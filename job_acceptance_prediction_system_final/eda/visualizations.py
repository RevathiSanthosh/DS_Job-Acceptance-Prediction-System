"""
eda/visualizations.py — 15 EDA charts covering all analyst tasks from the spec.
All saved as high-res PNG to outputs/eda_charts/.
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtk
import seaborn as sns
from config.settings import EDA_DIR, C

EDA_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "figure.facecolor":C["bg"],"axes.facecolor":C["card"],
    "axes.edgecolor":C["border"],"axes.labelcolor":C["muted"],
    "xtick.color":C["muted"],"ytick.color":C["muted"],
    "text.color":C["text"],"grid.color":C["border"],
    "grid.linestyle":"--","grid.linewidth":0.4,
    "font.family":"DejaVu Sans","axes.titlesize":13,
    "axes.titleweight":"bold","axes.titlecolor":C["primary"],
    "legend.facecolor":C["card"],"legend.edgecolor":C["border"],
})
PC = C["success"];  NC = C["danger"]
SPAL = {"Placed":PC,"Not Placed":NC}

def _sv(fig, name):
    p = EDA_DIR/name
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig); print(f"  💾 {name}"); return p

def _ax(ax, t="", xl="", yl=""):
    if t:  ax.set_title(t, color=C["primary"], pad=8)
    if xl: ax.set_xlabel(xl, color=C["muted"])
    if yl: ax.set_ylabel(yl, color=C["muted"])
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)

# 01 ── Placement Overview ────────────────────────────────────────
def c01_overview(df):
    placed = df["status"].value_counts()
    fig, axes = plt.subplots(1,2,figsize=(13,5),facecolor=C["bg"])
    fig.suptitle("📊 Job Placement Overview",fontsize=14,color=C["primary"],fontweight="bold")
    axes[0].pie(placed.values, labels=placed.index, autopct="%1.1f%%",
                colors=[PC,NC], wedgeprops={"edgecolor":C["bg"],"linewidth":2},
                pctdistance=0.82, startangle=90)
    axes[0].add_patch(plt.Circle((0,0),0.55,color=C["card"]))
    axes[0].set_title("Overall Placement Split",color=C["primary"])
    axes[0].set_facecolor(C["bg"])
    tr = df.groupby("company_tier")["status_num"].mean()*100
    clrs=[C["primary"],C["info"],C["accent"]]
    bars=axes[1].bar(tr.index,tr.values,color=clrs,edgecolor=C["bg"],width=0.5)
    for b in bars:
        h=b.get_height()
        axes[1].text(b.get_x()+b.get_width()/2,h+0.5,f"{h:.1f}%",
                     ha="center",va="bottom",fontsize=10,color=C["text"],fontweight="bold")
    _ax(axes[1],"Placement Rate by Company Tier","Company Tier","Rate (%)")
    axes[1].yaxis.set_major_formatter(mtk.PercentFormatter())
    fig.tight_layout(); return _sv(fig,"01_placement_overview.png")

# 02 ── Academic Scores ───────────────────────────────────────────
def c02_academic(df):
    fig,axes=plt.subplots(1,3,figsize=(17,5),facecolor=C["bg"])
    fig.suptitle("🎓 Academic Performance vs Placement",fontsize=14,color=C["primary"],fontweight="bold")
    for ax,col,ttl in zip(axes,["gpa","tenth_percentage","twelfth_percentage"],
                          ["GPA","10th %","12th %"]):
        for st,clr in SPAL.items():
            v=df[df["status"]==st][col].dropna()
            ax.hist(v,bins=25,alpha=0.65,color=clr,label=st,edgecolor=C["bg"])
            ax.axvline(v.median(),color=clr,lw=2,ls="--",alpha=0.9)
        _ax(ax,ttl,ttl,"Count"); ax.legend(fontsize=9)
    fig.tight_layout(); return _sv(fig,"02_academic_vs_placement.png")

# 03 ── Interview vs Placement ────────────────────────────────────
def c03_interview(df):
    fig,axes=plt.subplots(1,2,figsize=(14,5),facecolor=C["bg"])
    fig.suptitle("🎤 Interview & Employability vs Placement",fontsize=14,color=C["primary"],fontweight="bold")
    for ax,col in zip(axes,["interview_score","employability_score"]):
        for st,clr in SPAL.items():
            v=df[df["status"]==st][col].dropna()
            ax.hist(v,bins=25,alpha=0.65,color=clr,label=st,edgecolor=C["bg"])
            ax.axvline(v.median(),color=clr,lw=2,ls="--")
        _ax(ax,col.replace("_"," ").title(),col.replace("_"," ").title(),"Count")
        ax.legend(fontsize=9)
    fig.tight_layout(); return _sv(fig,"03_interview_vs_placement.png")

# 04 ── Skills Match ──────────────────────────────────────────────
def c04_skills(df):
    fig,axes=plt.subplots(1,2,figsize=(14,5),facecolor=C["bg"])
    fig.suptitle("💻 Skills Match Analysis",fontsize=14,color=C["primary"],fontweight="bold")
    pl=df[df["status"]=="Placed"]["skills_match_pct"].dropna()
    np2=df[df["status"]=="Not Placed"]["skills_match_pct"].dropna()
    bp=axes[0].boxplot([pl,np2],labels=["Placed","Not Placed"],patch_artist=True,notch=True,
                        medianprops={"color":C["accent"],"lw":2},
                        whiskerprops={"color":C["muted"]},capprops={"color":C["muted"]},
                        flierprops={"marker":".","alpha":0.3})
    bp["boxes"][0].set_facecolor(PC); bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor(NC); bp["boxes"][1].set_alpha(0.7)
    _ax(axes[0],"Skills Match % Boxplot","Status","Skills Match %")
    if "skills_level" in df.columns:
        sml=df.groupby(["skills_level","status"]).size().reset_index(name="n")
        piv=sml.pivot(index="skills_level",columns="status",values="n").fillna(0)
        piv=piv.reindex([l for l in ["Low","Medium","High","Excellent"] if l in piv.index])
        x=np.arange(len(piv)); w=0.35
        for i,(st,clr) in enumerate(SPAL.items()):
            if st in piv.columns:
                axes[1].bar(x+i*w,piv[st],w,label=st,color=clr,alpha=0.85)
        axes[1].set_xticks(x+w/2); axes[1].set_xticklabels(piv.index,rotation=20)
        axes[1].legend(); _ax(axes[1],"Placement by Skills Level","Skills Level","Count")
    fig.tight_layout(); return _sv(fig,"04_skills_analysis.png")

# 05 ── Company Tier ──────────────────────────────────────────────
def c05_tier(df):
    fig,axes=plt.subplots(1,2,figsize=(14,5),facecolor=C["bg"])
    fig.suptitle("🏢 Company Tier Analysis",fontsize=14,color=C["primary"],fontweight="bold")
    ta=df.groupby("company_tier")["status_num"].mean()*100
    clrs=[C["primary"],C["info"],C["accent"]]
    bars=axes[0].bar(ta.index,ta.values,color=clrs,edgecolor=C["bg"],width=0.5)
    for b in bars:
        h=b.get_height()
        axes[0].text(b.get_x()+b.get_width()/2,h+0.5,f"{h:.1f}%",
                     ha="center",va="bottom",color=C["text"],fontsize=10,fontweight="bold")
    _ax(axes[0],"Acceptance Rate by Tier","Tier","Rate %")
    axes[0].yaxis.set_major_formatter(mtk.PercentFormatter())
    tc=df.groupby(["company_tier","status"]).size().unstack(fill_value=0)
    tc.plot(kind="bar",ax=axes[1],color=[PC,NC],edgecolor=C["bg"],width=0.6)
    axes[1].set_xticklabels(tc.index,rotation=15); _ax(axes[1],"Volume by Tier","Tier","Count")
    axes[1].legend()
    fig.tight_layout(); return _sv(fig,"05_company_tier.png")

# 06 ── Experience ────────────────────────────────────────────────
def c06_experience(df):
    fig,axes=plt.subplots(1,2,figsize=(14,5),facecolor=C["bg"])
    fig.suptitle("📅 Work Experience vs Placement",fontsize=14,color=C["primary"],fontweight="bold")
    for st,clr in SPAL.items():
        v=df[df["status"]==st]["years_experience"].dropna()
        axes[0].hist(v,bins=20,alpha=0.65,color=clr,label=st,edgecolor=C["bg"])
    _ax(axes[0],"Years Experience Distribution","Years","Count"); axes[0].legend()
    if "experience_category" in df.columns:
        ea=df.groupby("experience_category")["status_num"].mean()*100
        order=[o for o in ["Fresher","Junior","Mid-Level","Senior"] if o in ea.index]
        ea=ea.reindex(order)
        axes[1].bar(ea.index,ea.values,color=[C["primary"],C["info"],C["success"],C["accent"]],
                    edgecolor=C["bg"],width=0.5)
        for i,(idx,val) in enumerate(ea.items()):
            axes[1].text(i,val+0.5,f"{val:.1f}%",ha="center",va="bottom",
                         color=C["text"],fontsize=10,fontweight="bold")
        _ax(axes[1],"Placement Rate by Experience","Category","Rate %")
        axes[1].yaxis.set_major_formatter(mtk.PercentFormatter())
    fig.tight_layout(); return _sv(fig,"06_experience.png")

# 07 ── Competition Level ─────────────────────────────────────────
def c07_competition(df):
    fig,axes=plt.subplots(1,2,figsize=(14,5),facecolor=C["bg"])
    fig.suptitle("⚔️  Competition Level vs Acceptance",fontsize=14,color=C["primary"],fontweight="bold")
    ca=df.groupby("competition_level")["status_num"].mean()*100
    order=[c for c in ["Low","Medium","High"] if c in ca.index]
    ca=ca.reindex(order)
    clrs=[C["success"],C["accent"],C["danger"]]
    bars=axes[0].bar(ca.index,ca.values,color=clrs,edgecolor=C["bg"],width=0.5)
    for b in bars:
        h=b.get_height()
        axes[0].text(b.get_x()+b.get_width()/2,h+0.5,f"{h:.1f}%",
                     ha="center",va="bottom",color=C["text"],fontsize=10)
    _ax(axes[0],"Acceptance Rate by Competition","Competition","Rate %")
    axes[0].yaxis.set_major_formatter(mtk.PercentFormatter())
    cc=df.groupby(["competition_level","status"]).size().unstack(fill_value=0)
    cc=cc.reindex([c for c in ["Low","Medium","High"] if c in cc.index])
    cc.plot(kind="bar",ax=axes[1],color=[PC,NC],edgecolor=C["bg"])
    axes[1].set_xticklabels(cc.index,rotation=0)
    _ax(axes[1],"Volume by Competition","Competition","Count"); axes[1].legend()
    fig.tight_layout(); return _sv(fig,"07_competition.png")

# 08 ── Certification Impact ──────────────────────────────────────
def c08_certification(df):
    fig,axes=plt.subplots(1,2,figsize=(14,5),facecolor=C["bg"])
    fig.suptitle("🏅 Certification Impact",fontsize=14,color=C["primary"],fontweight="bold")
    cr=df.groupby("certification")["status_num"].mean().sort_values(ascending=False)*100
    clrs=plt.cm.plasma(np.linspace(0.2,0.9,len(cr)))
    axes[0].barh(cr.index[::-1],cr.values[::-1],color=clrs,edgecolor=C["bg"])
    _ax(axes[0],"Acceptance Rate by Certification")
    axes[0].set_xlabel("Acceptance Rate %",color=C["muted"])
    axes[0].xaxis.set_major_formatter(mtk.PercentFormatter())
    if "has_certification" in df.columns:
        cg=df.groupby("has_certification")["status_num"].mean()*100
        axes[1].bar(["No Cert","Has Cert"],cg.values,
                    color=[NC,PC],edgecolor=C["bg"],width=0.45)
        for i,val in enumerate(cg.values):
            axes[1].text(i,val+0.5,f"{val:.1f}%",ha="center",va="bottom",
                         color=C["text"],fontsize=12,fontweight="bold")
        _ax(axes[1],"Cert vs No Cert","Status","Rate %")
        axes[1].yaxis.set_major_formatter(mtk.PercentFormatter())
    fig.tight_layout(); return _sv(fig,"08_certification.png")

# 09 ── Correlation Heatmap ───────────────────────────────────────
def c09_correlation(df):
    cols=["gpa","tenth_percentage","twelfth_percentage","years_experience",
          "skills_match_pct","interview_score","employability_score",
          "aptitude_score","backlogs","status_num","placement_probability"]
    cols=[c for c in cols if c in df.columns]
    corr=df[cols].corr()
    fig,ax=plt.subplots(figsize=(12,9),facecolor=C["bg"])
    fig.suptitle("🔥 Feature Correlation Heatmap",fontsize=14,color=C["primary"],fontweight="bold")
    mask=np.triu(np.ones_like(corr,dtype=bool))
    sns.heatmap(corr,mask=mask,annot=True,fmt=".2f",cmap="coolwarm",
                linewidths=0.5,linecolor=C["bg"],ax=ax,
                cbar_kws={"shrink":0.8},annot_kws={"size":9})
    ax.tick_params(axis="x",rotation=35)
    fig.tight_layout(); return _sv(fig,"09_correlation.png")

# 10 ── Placement Probability ─────────────────────────────────────
def c10_prob(df):
    if "placement_probability" not in df.columns: return
    fig,axes=plt.subplots(1,2,figsize=(14,5),facecolor=C["bg"])
    fig.suptitle("🎯 Placement Probability Score",fontsize=14,color=C["primary"],fontweight="bold")
    for st,clr in SPAL.items():
        v=df[df["status"]==st]["placement_probability"].dropna()
        axes[0].hist(v,bins=30,alpha=0.65,color=clr,label=st,edgecolor=C["bg"])
    _ax(axes[0],"Distribution by Outcome","Score","Count"); axes[0].legend()
    pl=df[df["status"]=="Placed"]["placement_probability"].dropna()
    np2=df[df["status"]=="Not Placed"]["placement_probability"].dropna()
    parts=axes[1].violinplot([pl,np2],positions=[1,2],showmedians=True)
    for pc2,clr in zip(parts["bodies"],[PC,NC]):
        pc2.set_facecolor(clr); pc2.set_alpha(0.6)
    parts["cmedians"].set_color(C["accent"]); parts["cmedians"].set_linewidth(2)
    axes[1].set_xticks([1,2]); axes[1].set_xticklabels(["Placed","Not Placed"])
    _ax(axes[1],"Score Violin Plot","","Score")
    fig.tight_layout(); return _sv(fig,"10_placement_probability.png")

# 11 ── Backlogs ──────────────────────────────────────────────────
def c11_backlogs(df):
    fig,ax=plt.subplots(figsize=(11,5),facecolor=C["bg"])
    fig.suptitle("📚 Backlogs vs Placement Rate",fontsize=14,color=C["primary"],fontweight="bold")
    bl=df.groupby("backlogs")["status_num"].mean()*100
    clrs=[C["success"] if i==0 else (C["accent"] if i==1 else C["danger"]) for i in bl.index]
    bars=ax.bar(bl.index.astype(str),bl.values,color=clrs,edgecolor=C["bg"],width=0.6)
    for b in bars:
        h=b.get_height()
        ax.text(b.get_x()+b.get_width()/2,h+0.4,f"{h:.1f}%",ha="center",
                va="bottom",color=C["text"],fontsize=10)
    _ax(ax,"Placement Rate vs Number of Backlogs","Backlogs","Rate %")
    ax.yaxis.set_major_formatter(mtk.PercentFormatter())
    fig.tight_layout(); return _sv(fig,"11_backlogs.png")

# 12 ── Salary Analysis ───────────────────────────────────────────
def c12_salary(df):
    fig,axes=plt.subplots(1,2,figsize=(14,5),facecolor=C["bg"])
    fig.suptitle("💰 Salary Analysis",fontsize=14,color=C["primary"],fontweight="bold")
    for st,clr in SPAL.items():
        v=df[df["status"]==st]["offered_salary_inr"].dropna()/100000
        axes[0].hist(v,bins=25,alpha=0.65,color=clr,label=st,edgecolor=C["bg"])
    _ax(axes[0],"Salary Distribution","Salary (₹ Lakhs)","Count"); axes[0].legend()
    if "salary_band" in df.columns:
        sa=df.groupby("salary_band")["status_num"].mean()*100
        order=[o for o in ["<₹5L","₹5-9L","₹9-13L",">₹13L"] if o in sa.index]
        sa=sa.reindex(order)
        axes[1].bar(sa.index,sa.values,
                    color=[C["accent"],C["info"],C["primary"],C["success"]],
                    edgecolor=C["bg"],width=0.5)
        for i,val in enumerate(sa.values):
            axes[1].text(i,val+0.5,f"{val:.1f}%",ha="center",va="bottom",color=C["text"])
        _ax(axes[1],"Acceptance Rate by Salary Band","Band","Rate %")
        axes[1].yaxis.set_major_formatter(mtk.PercentFormatter())
    fig.tight_layout(); return _sv(fig,"12_salary.png")

# 13 ── Gender Analysis ───────────────────────────────────────────
def c13_gender(df):
    fig,axes=plt.subplots(1,2,figsize=(14,5),facecolor=C["bg"])
    fig.suptitle("👤 Gender & Demographic Analysis",fontsize=14,color=C["primary"],fontweight="bold")
    ga=df.groupby("gender")["status_num"].mean()*100
    axes[0].bar(ga.index,ga.values,color=[C["primary"],C["info"],C["accent"]],
                edgecolor=C["bg"],width=0.45)
    for i,val in enumerate(ga.values):
        axes[0].text(i,val+0.5,f"{val:.1f}%",ha="center",va="bottom",
                     color=C["text"],fontsize=11,fontweight="bold")
    _ax(axes[0],"Placement Rate by Gender","Gender","Rate %")
    gc=df["gender"].value_counts()
    axes[1].pie(gc.values,labels=gc.index,autopct="%1.1f%%",
                colors=[C["primary"],C["info"],C["accent"]],
                wedgeprops={"edgecolor":C["bg"]},pctdistance=0.80,startangle=90)
    axes[1].add_patch(plt.Circle((0,0),0.55,color=C["card"]))
    axes[1].set_title("Gender Distribution",color=C["primary"])
    axes[1].set_facecolor(C["bg"])
    fig.tight_layout(); return _sv(fig,"13_gender.png")

# 14 ── Job Role ──────────────────────────────────────────────────
def c14_jobrole(df):
    fig,axes=plt.subplots(1,2,figsize=(16,6),facecolor=C["bg"])
    fig.suptitle("💼 Job Role Analysis",fontsize=14,color=C["primary"],fontweight="bold")
    ra=df.groupby("job_role")["status_num"].mean().sort_values()*100
    axes[0].barh(ra.index,ra.values,color=plt.cm.plasma(np.linspace(0.2,0.9,len(ra))),edgecolor=C["bg"])
    _ax(axes[0],"Placement Rate by Job Role","Rate %","")
    axes[0].xaxis.set_major_formatter(mtk.PercentFormatter())
    rc=df["job_role"].value_counts().head(8)
    axes[1].barh(rc.index[::-1],rc.values[::-1],
                 color=plt.cm.cool(np.linspace(0.2,0.9,len(rc))),edgecolor=C["bg"])
    _ax(axes[1],"Applications by Job Role","Candidates","")
    fig.tight_layout(); return _sv(fig,"14_jobroles.png")

# 15 ── Academic Band ─────────────────────────────────────────────
def c15_academicband(df):
    if "academic_band" not in df.columns: return
    fig,ax=plt.subplots(figsize=(11,5),facecolor=C["bg"])
    fig.suptitle("🎓 Academic Band vs Placement Rate",fontsize=14,color=C["primary"],fontweight="bold")
    ab=df.groupby("academic_band")["status_num"].mean()*100
    order=[o for o in ["Poor","Below Avg","Average","Good","Excellent"] if o in ab.index]
    ab=ab.reindex(order)
    clrs=[C["danger"],C["accent"],C["info"],C["primary"],C["success"]]
    bars=ax.bar(ab.index,ab.values,color=clrs[:len(ab)],edgecolor=C["bg"],width=0.55)
    for b in bars:
        h=b.get_height()
        ax.text(b.get_x()+b.get_width()/2,h+0.4,f"{h:.1f}%",ha="center",
                va="bottom",color=C["text"],fontsize=10,fontweight="bold")
    _ax(ax,"Placement Rate by Academic Band","Academic Band","Rate %")
    ax.yaxis.set_major_formatter(mtk.PercentFormatter())
    fig.tight_layout(); return _sv(fig,"15_academic_band.png")


def run_all(df: pd.DataFrame) -> list:
    print("\n"+"═"*55+"\n  📊  EDA VISUALIZATIONS\n"+"═"*55)
    fns=[c01_overview,c02_academic,c03_interview,c04_skills,c05_tier,
         c06_experience,c07_competition,c08_certification,c09_correlation,
         c10_prob,c11_backlogs,c12_salary,c13_gender,c14_jobrole,c15_academicband]
    paths=[]
    for fn in fns:
        try:
            p=fn(df)
            if p: paths.append(p)
        except Exception as e:
            print(f"  ⚠️  {fn.__name__}: {e}")
    print(f"\n  ✅ {len(paths)}/15 EDA charts saved → {EDA_DIR}\n")
    return paths
