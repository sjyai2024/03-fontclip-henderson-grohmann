import io
import re
import sys
import math
import json
import hashlib
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="03B2 FontCLIP Henderson–Grohmann", layout="wide")

APP_VERSION="1.2"
APP_DIR=Path(__file__).resolve().parent
RUNTIME_ROOT=Path.home()/".cache"/"fontclip_brand_personality_03b2"

FONTCLIP_COMMIT="3d4c6af01f668800d8e4f9f4f753d29c74dad252"
FONTCLIP_ARCHIVE_URL=f"https://github.com/yukistavailable/FontCLIP/archive/{FONTCLIP_COMMIT}.zip"
FONTCLIP_CHECKPOINT_GDRIVE_ID="1Tym7rAIuaGr6Gv-gZRSJmPstQjOWPgl1"
EXPECTED_CHECKPOINT_SHA256="c441277fbed4366d32d8fb65725189b97d3fe88bae5fe0648b969feea01bbb00"

REVIEW_FILE=APP_DIR/"03A_eligibility_review_final.csv"
COEF_FILE=APP_DIR/"Grohmann2013_standardized_coefficients.csv"
PROMPT_FILE=APP_DIR/"Henderson_Grohmann_FontCLIP_prompts.csv"
TEXT_FILES={
    "MiniLM":APP_DIR/"02B_MiniLM_5D_raw.csv",
    "mDeBERTa":APP_DIR/"02B_mDeBERTa_5D_raw.csv",
}
FEATURES=["Elaborate","Harmony","Natural","Weight","Flourish"]
DIMENSIONS=["Sincerity","Excitement","Competence","Sophistication","Ruggedness"]
IMAGE_EXTS=(".png",".jpg",".jpeg",".webp")

def slug(s):
    return re.sub(r"[^a-z0-9]+","_",str(s).lower()).strip("_")

def brand_key(s):
    s=str(s).lower().replace("ö","o")
    return re.sub(r"[^a-z0-9]","",s)

def basename_key(s):
    return Path(str(s)).name.lower()

def sha256_file(path,chunk_size=1024*1024):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        while True:
            b=f.read(chunk_size)
            if not b: break
            h.update(b)
    return h.hexdigest()

def download_file(url,path,timeout=120):
    path.parent.mkdir(parents=True,exist_ok=True)
    with requests.get(url,stream=True,timeout=timeout) as r:
        r.raise_for_status()
        with open(path,"wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk: f.write(chunk)

def prepare_fontclip_source():
    RUNTIME_ROOT.mkdir(parents=True,exist_ok=True)
    source_root=RUNTIME_ROOT/"source"/f"FontCLIP-{FONTCLIP_COMMIT}"
    if source_root.exists() and (source_root/"models").exists():
        return source_root
    archive_path=RUNTIME_ROOT/f"fontclip-{FONTCLIP_COMMIT}.zip"
    if not archive_path.exists():
        download_file(FONTCLIP_ARCHIVE_URL,archive_path)
    parent=RUNTIME_ROOT/"source"; parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(parent)
    candidates=[p for p in parent.iterdir() if p.is_dir() and p.name.startswith("FontCLIP-") and (p/"models").exists()]
    if not candidates:
        raise RuntimeError("FontCLIP source archive structure could not be recognized.")
    return candidates[0]

def prepare_checkpoint():
    ckpt=RUNTIME_ROOT/"model_checkpoints"/"model.pt"
    if ckpt.exists() and ckpt.stat().st_size>10*1024*1024:
        if sha256_file(ckpt)==EXPECTED_CHECKPOINT_SHA256:
            return ckpt
        ckpt.unlink(missing_ok=True)
    ckpt.parent.mkdir(parents=True,exist_ok=True)
    import gdown
    url=f"https://drive.google.com/uc?id={FONTCLIP_CHECKPOINT_GDRIVE_ID}"
    out=gdown.download(url,str(ckpt),quiet=False)
    if not out or not ckpt.exists() or ckpt.stat().st_size<=10*1024*1024:
        raise RuntimeError("Official FontCLIP checkpoint download failed.")
    actual=sha256_file(ckpt)
    if actual!=EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"Checkpoint SHA256 mismatch: {actual}")
    return ckpt

@st.cache_resource(show_spinner=False)
def load_fontclip_runtime():
    # FontCLIP's upstream tokenizer imports pkg_resources, which is supplied
    # by setuptools. Keep this explicit because some Streamlit Cloud images
    # omit it from the runtime environment.
    try:
        import pkg_resources  # noqa: F401
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "FontCLIP 실행에 필요한 `pkg_resources`가 없습니다. "
            "requirements.txt에 `setuptools==80.9.0`가 포함되어 있는지 확인한 뒤 "
            "Streamlit 앱을 Reboot 하세요."
        ) from e

    repo_dir=prepare_fontclip_source()
    checkpoint_path=prepare_checkpoint()
    if str(repo_dir) not in sys.path:
        sys.path.insert(0,str(repo_dir))
    from models.init_model import device,load_model,preprocess
    from models.lora import LoRAConfig
    from utils.tokenizer import tokenize
    lora_config_text=LoRAConfig(
        r=256,alpha=1024.0,bias=False,learnable_alpha=False,
        apply_q=True,apply_k=True,apply_v=True,apply_out=True
    )
    model=load_model(
        str(checkpoint_path),model_name="ViT-B/32",
        use_oft_vision=False,use_oft_text=False,
        oft_config_vision=None,oft_config_text=None,
        use_lora_text=True,use_lora_vision=False,
        lora_config_vision=None,lora_config_text=lora_config_text,
        use_coop_text=False,use_coop_vision=False,
        precontext_length_vision=10,precontext_length_text=77,
        precontext_dropout_rate=0,pt_applied_layers=None,
    )
    model.eval()
    meta={
        "FontCLIP_commit":FONTCLIP_COMMIT,
        "Checkpoint_SHA256":sha256_file(checkpoint_path),
        "Model":"FontCLIP / ViT-B/32 / LoRA-text checkpoint",
    }
    return model,preprocess,tokenize,device,meta

def safe_extract_zip(uploaded_bytes):
    out=[]
    with zipfile.ZipFile(io.BytesIO(uploaded_bytes)) as zf:
        for info in zf.infolist():
            if info.is_dir() or info.filename.startswith("__MACOSX/"): continue
            if not info.filename.lower().endswith(IMAGE_EXTS): continue
            data=zf.read(info)
            img=Image.open(io.BytesIO(data)); img.load()
            out.append({"filename":info.filename,"image":img})
    return out

def merge_review(items,review):
    meta=pd.DataFrame([{"Filename":x["filename"]} for x in items])
    meta["_base"]=meta["Filename"].map(basename_key)
    rv=review.copy()
    rv["_base"]=rv["Filename"].map(basename_key)
    keep=["_base","Brand","Eligibility","Researcher_Note"]
    meta=meta.merge(rv[keep].drop_duplicates("_base"),on="_base",how="left")
    meta["Brand"]=meta["Brand"].fillna(meta["Filename"].map(lambda x:Path(x).stem))
    meta["Eligibility"]=meta["Eligibility"].fillna("Unreviewed")
    meta["Researcher_Note"]=meta["Researcher_Note"].fillna("")
    # preserve image order by filename mapping
    imap={x["filename"]:x["image"] for x in items}
    meta["Image"]=[imap[x] for x in meta["Filename"]]
    return meta.drop(columns="_base")

def encode_prompt_scores(images,prompts,model,preprocess,tokenize,device,batch_size=8):
    import torch
    tokens=tokenize(prompts).to(device)
    with torch.no_grad():
        txt=model.encode_text(tokens).float()
        txt=txt/txt.norm(dim=-1,keepdim=True)
    scores=[]
    for s in range(0,len(images),batch_size):
        tensors=[preprocess(im.convert("RGB")) for im in images[s:s+batch_size]]
        x=torch.stack(tensors).to(device)
        with torch.no_grad():
            feat=model.encode_image(x).float()
            feat=feat/feat.norm(dim=-1,keepdim=True)
            sim=feat@txt.T
        scores.append(sim.cpu().numpy())
    return np.vstack(scores)

def feature_scores(prompt_scores,prompt_map,meta):
    out=meta[["Filename","Brand","Eligibility","Researcher_Note"]].copy().reset_index(drop=True)
    for feat in FEATURES:
        idx=prompt_map.index[prompt_map["Typeface_Feature"].eq(feat)].tolist()
        out[feat]=prompt_scores[:,idx].mean(axis=1)
    return out

def z_by_reference(df,reference_rule):
    out=df.copy()
    if reference_rule=="A only":
        ref=out[out["Eligibility"].eq("A")]
    else:
        ref=out[out["Eligibility"].isin(["A","B"])]
    stats=[]
    for f in FEATURES:
        mu=float(ref[f].mean()); sd=float(ref[f].std(ddof=1))
        if sd<1e-12: raise ValueError(f"Reference SD=0 for {f}")
        out[f+"__Z"]=(out[f]-mu)/sd
        stats.append({"Feature":f,"Reference_Mean":mu,"Reference_SD":sd,"Reference_N":len(ref)})
    return out,pd.DataFrame(stats)

def apply_grohmann(zdf,coef,mapping_name):
    C=coef[coef["Mapping"].eq(mapping_name)].copy()
    out=zdf[["Filename","Brand","Eligibility","Researcher_Note"]+[f+"__Z" for f in FEATURES]].copy()
    for d in DIMENSIONS:
        g=C[C["Dimension"].eq(d)].set_index("Typeface_Feature")
        out[d]=0.0
        for f in FEATURES:
            out[d]+=out[f+"__Z"]*float(g.loc[f,"Standardized_Beta"])
    return out

def z_columns(df,cols):
    out=df.copy()
    for c in cols:
        mu=out[c].mean(); sd=out[c].std(ddof=1)
        out[c]=(out[c]-mu)/sd
    return out

def profile_corr(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    if np.std(a)<1e-12 or np.std(b)<1e-12: return np.nan
    return float(np.corrcoef(a,b)[0,1])

def align_common(text,visual):
    T=text.copy(); V=visual.copy()
    T["_key"]=T["Brand"].map(brand_key); V["_key"]=V["Brand"].map(brand_key)
    common=sorted(set(T["_key"])&set(V["_key"]))
    T=T[T["_key"].isin(common)].set_index("_key").loc[common].reset_index()
    V=V[V["_key"].isin(common)].set_index("_key").loc[common].reset_index()
    # z-standardize each dimension inside each modality across the exact common set
    Tz=z_columns(T[["Brand"]+DIMENSIONS],DIMENSIONS)
    Vz=z_columns(V[["Brand"]+DIMENSIONS],DIMENSIONS)
    return common,T,V,Tz,Vz

def distance_matrix(A,B):
    a=A[DIMENSIONS].to_numpy(float); b=B[DIMENSIONS].to_numpy(float)
    return np.sqrt(((a[:,None,:]-b[None,:,:])**2).sum(axis=2))

def corr_matrix(A,B):
    a=A[DIMENSIONS].to_numpy(float); b=B[DIMENSIONS].to_numpy(float)
    M=np.empty((len(a),len(b)))
    for i in range(len(a)):
        for j in range(len(b)):
            M[i,j]=profile_corr(a[i],b[j])
    return M

def permutation_distance(D,n_perm=10000,seed=20260927):
    n=D.shape[0]
    observed=float(np.mean(np.diag(D)))
    off=float(np.mean(D[~np.eye(n,dtype=bool)]))
    rng=np.random.default_rng(seed)
    vals=np.empty(n_perm)
    for k in range(n_perm):
        p=rng.permutation(n)
        vals[k]=np.mean([D[i,p[i]] for i in range(n)])
    p_small=(np.sum(vals<=observed)+1)/(n_perm+1)
    return observed,off,p_small

def permutation_corr(M,n_perm=10000,seed=20260927):
    n=M.shape[0]
    observed=float(np.nanmean(np.diag(M)))
    off=float(np.nanmean(M[~np.eye(n,dtype=bool)]))
    rng=np.random.default_rng(seed)
    vals=np.empty(n_perm)
    for k in range(n_perm):
        p=rng.permutation(n)
        vals[k]=np.nanmean([M[i,p[i]] for i in range(n)])
    p_large=(np.sum(vals>=observed)+1)/(n_perm+1)
    return observed,off,p_large

def retrieval_table(common,T,V,D):
    rows=[]
    for i,key in enumerate(common):
        order=np.argsort(D[i])
        rank=int(np.where(order==i)[0][0])+1
        rows.append({
            "Brand_Text":T.iloc[i]["Brand"],
            "Brand_Visual":V.iloc[i]["Brand"],
            "Matched_Distance":float(D[i,i]),
            "Retrieval_Rank":rank,
            "Top1":int(rank==1),
            "Top3":int(rank<=3),
            "Top5":int(rank<=5),
        })
    return pd.DataFrame(rows)

def evaluate_h1(text,visual,n_perm):
    common,T,V,Tz,Vz=align_common(text,visual)
    D=distance_matrix(Tz,Vz)
    M=corr_matrix(Tz,Vz)
    dmatch,doff,p_dist=permutation_distance(D,n_perm)
    rmatch,roff,pr=permutation_corr(M,n_perm)
    retr=retrieval_table(common,T,V,D)
    summary=pd.DataFrame([{
        "N_Common":len(common),
        "Matched_Mean_Distance":dmatch,
        "Nonmatching_Mean_Distance":doff,
        "Distance_Permutation_p_one_sided":p_dist,
        "Matched_Mean_Profile_r":rmatch,
        "Nonmatching_Mean_Profile_r":roff,
        "Correlation_Permutation_p_one_sided":pr,
        "Mean_Retrieval_Rank":retr["Retrieval_Rank"].mean(),
        "Median_Retrieval_Rank":retr["Retrieval_Rank"].median(),
        "Top1_Rate":retr["Top1"].mean(),
        "Top3_Rate":retr["Top3"].mean(),
        "Top5_Rate":retr["Top5"].mean(),
        "Random_Expected_Mean_Rank":(len(common)+1)/2,
    }])
    return summary,retr,Tz,Vz

def zip_csv(files,meta):
    bio=io.BytesIO()
    with zipfile.ZipFile(bio,"w",zipfile.ZIP_DEFLATED) as zf:
        for name,df in files.items():
            zf.writestr(name,df.to_csv(index=False).encode("utf-8-sig"))
        zf.writestr("03B2_99_run_metadata.json",json.dumps(meta,ensure_ascii=False,indent=2))
    return bio.getvalue()

# UI
st.title(f"03B2 · FontCLIP → Henderson/Grohmann → Aaker 5D · v{APP_VERSION}")
st.caption("FontCLIP이 Aaker 개성을 직접 측정하지 않고, 서체 조형특성을 먼저 측정한 뒤 Grohmann et al. (2013)의 외부 고정계수로 Aaker 5D를 도출합니다.")

st.info(
    "Primary mapping은 Grohmann Study 3의 표준화 회귀계수입니다. "
    "Study 3은 여러 브랜드·여러 서체를 함께 평가한 multi-brand context이며 color를 통제했고 black이 baseline입니다. "
    "본 연구에서는 새 회귀계수를 학습하지 않습니다."
)

with st.expander("사전고정된 FontCLIP 서체특성 프롬프트",expanded=True):
    st.dataframe(pd.read_csv(PROMPT_FILE),use_container_width=True,hide_index=True)

with st.expander("Grohmann 고정계수",expanded=False):
    st.dataframe(pd.read_csv(COEF_FILE),use_container_width=True,hide_index=True)

logo_zip=st.file_uploader("`logo_dataset_68_fixed.zip` 업로드",type=["zip"])
reference_rule=st.radio("Font feature reference",["A only","A+B"],index=0,horizontal=True)
nperm=st.selectbox("Permutation 횟수",[1000,5000,10000,50000],index=2)

if st.button("03B2 분석 실행",type="primary",disabled=logo_zip is None):
    try:
        review=pd.read_csv(REVIEW_FILE)
        prompt_map=pd.read_csv(PROMPT_FILE)
        coef=pd.read_csv(COEF_FILE)
        items=safe_extract_zip(logo_zip.getvalue())
        meta=merge_review(items,review)

        model,preprocess,tokenize,device,model_meta=load_fontclip_runtime()
        prompts=prompt_map["Prompt"].tolist()
        with st.spinner("FontCLIP으로 Henderson/Grohmann 서체특성 평가"):
            S=encode_prompt_scores(meta["Image"].tolist(),prompts,model,preprocess,tokenize,device)
        feat=feature_scores(S,prompt_map,meta.drop(columns="Image"))
        feat_z,feat_stats=z_by_reference(feat,reference_rule)

        v3=apply_grohmann(feat_z,coef,"Study3_primary")
        v1=apply_grohmann(feat_z,coef,"Study1_sensitivity")

        # Primary eligibility for H1 remains A only regardless of sensitivity reference option.
        v3A=v3[v3["Eligibility"].eq("A")].copy()
        v1A=v1[v1["Eligibility"].eq("A")].copy()

        results={}
        files={
            "03B2_01_typeface_feature_raw.csv":feat,
            "03B2_02_typeface_feature_z.csv":feat_z,
            "03B2_03_reference_stats.csv":feat_stats,
            "03B2_04_Aaker5D_Study3_primary.csv":v3,
            "03B2_05_Aaker5D_Study1_sensitivity.csv":v1,
        }

        summaries=[]
        for text_label,path in TEXT_FILES.items():
            text=pd.read_csv(path)
            for map_label,visual in [("Study3_primary",v3A),("Study1_sensitivity",v1A)]:
                sm,retr,tz,vz=evaluate_h1(text,visual,nperm)
                sm.insert(0,"Text_Model",text_label)
                sm.insert(1,"Typeface_Mapping",map_label)
                summaries.append(sm)
                files[f"03B2_H1_{text_label}_{map_label}_retrieval.csv"]=retr
                files[f"03B2_H1_{text_label}_{map_label}_text_z.csv"]=tz
                files[f"03B2_H1_{text_label}_{map_label}_visual_z.csv"]=vz
        summary=pd.concat(summaries,ignore_index=True)
        files["03B2_06_H1_summary.csv"]=summary

        st.session_state["03B2"]={
            "files":files,"summary":summary,"device":str(device),
            "model_meta":model_meta,"reference_rule":reference_rule,"nperm":nperm,
        }
        st.success("03B2 분석 완료")
    except Exception as e:
        st.exception(e)

if "03B2" in st.session_state:
    R=st.session_state["03B2"]
    st.header("H1 · 동일브랜드가 비일치 브랜드보다 가까운가?")
    st.dataframe(
        R["summary"].style.format({
            "Matched_Mean_Distance":"{:.3f}",
            "Nonmatching_Mean_Distance":"{:.3f}",
            "Distance_Permutation_p_one_sided":"{:.4f}",
            "Matched_Mean_Profile_r":"{:.3f}",
            "Nonmatching_Mean_Profile_r":"{:.3f}",
            "Correlation_Permutation_p_one_sided":"{:.4f}",
            "Mean_Retrieval_Rank":"{:.2f}",
            "Median_Retrieval_Rank":"{:.1f}",
            "Top1_Rate":"{:.3f}","Top3_Rate":"{:.3f}","Top5_Rate":"{:.3f}",
            "Random_Expected_Mean_Rank":"{:.1f}",
        }),
        use_container_width=True,hide_index=True
    )
    st.caption(
        "Primary H1 criterion: Study3_primary × MiniLM에서 matched mean distance가 "
        "nonmatching보다 작고 one-sided permutation p<.05인지 확인합니다. "
        "mDeBERTa와 Study1은 robustness/sensitivity입니다."
    )

    meta={
        "App_Version":APP_VERSION,
        **R["model_meta"],
        "Feature_Prompts":"published-definition-derived ensemble",
        "Feature_Reference":R["reference_rule"],
        "Primary_Mapping":"Grohmann et al. 2013 Study 3 standardized coefficients",
        "Sensitivity_Mapping":"Grohmann et al. 2013 Study 1 standardized coefficients",
        "Primary_Text_Model":"MiniLM (kept because it was preregistered primary before results)",
        "Robustness_Text_Model":"mDeBERTa",
        "Primary_H1_Metric":"matched Euclidean distance < mismatched after within-modality column z-standardization on common brands",
        "Permutation_N":R["nperm"],
        "Dimensions":DIMENSIONS,
    }
    data=zip_csv(R["files"],meta)
    st.download_button(
        "03B2 전체 결과 ZIP 다운로드",
        data=data,
        file_name="03B2_fontclip_henderson_grohmann_results_v1_0.zip",
        mime="application/zip"
    )
