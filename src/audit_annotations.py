# /// script
# requires-python = ">=3.13"
# dependencies = ["pyarrow>=18"]
# ///
"""Gold-corpus annotation convention audit and label-context profile."""
from __future__ import annotations
import collections, json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_probe import reconstruct
import pyarrow.parquet as pq
OUT = ROOT / "output/data_understanding"
HON = re.compile(r"^(?:mr|mrs|ms|miss|dr|prof|rev|sir|madam)\.?(?:\s|$)", re.I)
HON_BEFORE = re.compile(r"\b(?:mr|mrs|ms|miss|dr|prof|rev|sir|madam)\.?\s+$", re.I)
NUM = re.compile(r"(?:\d|#|@|/|-{1,2}|\b(?:ref|id|user|acct|ticket|case)\b)", re.I)
PUN = re.compile(r"[^\w\s]", re.UNICODE)
def pct(n,d): return round(100*n/d,4) if d else 0.0
def ctx(t,a,b,r=42):
    lo,hi=max(0,a-r),min(len(t),b+r)
    return {"before":t[lo:a],"span":t[a:b],"after":t[b:hi],"start":a,"end":b}
def reg(t,a,b):
    ls=t.rfind("\n",0,a)+1; le=t.find("\n",b); le=len(t) if le<0 else le
    line=t[ls:le]; before=t[:a]; after=t[b:]
    if a<=max(1,int(len(t)*.12)) and re.search(r"(?:from|to|subject|date|cc)\s*:",line,re.I): return "header"
    if re.search(r"(?:regards|sincerely|thank you|best),?\s*$",before[-160:],re.I) or (a>int(len(t)*.72) and len(line.split())<=10): return "signature_or_tail"
    if a<int(len(t)*.18): return "opening"
    if "\n\n" in before[-4:] or "\n\n" in after[:4]: return "paragraph_boundary"
    return "body"
def main():
    texts,gold=reconstruct()
    meta={r["document_id"]:r for r in pq.read_table(ROOT/"Data/train/train_metadata.parquet").to_pylist()}
    labels=sorted({l for ss in gold.values() for _,_,l in ss})
    totals={"documents":len(texts),"annotated_documents":sum(bool(x) for x in gold.values()),"annotations":sum(map(len,gold.values())),"labels":labels}
    bd=collections.defaultdict(lambda:collections.Counter()); lp=collections.defaultdict(collections.Counter); cd=collections.defaultdict(collections.Counter); rc=collections.defaultdict(collections.Counter); ex=collections.defaultdict(list); cross=collections.defaultdict(collections.Counter)
    targets={"Medical Clinic of Anytown","10:00 AM","Attorney Robert Davis","Robert Davis"}
    rep={"annotated_repeated_values":0,"unlabelled_exact_occurrences":0,"overlapping_unlabelled_occurrences":0,"free_unlabelled_occurrences":0,"documents_with_repeated_value":0,"documents_with_unlabelled_repeat":0}
    unlabelled_docs=set()
    for doc,spans in gold.items():
        t=texts[doc]; m=meta[doc]; vals=collections.defaultdict(list); seen=set()
        for a,b,l in spans:
            v=t[a:b]; vals[v].append((a,b,l)); seen.add(v); c=bd[l]; c["count"]+=1
            if v in targets and len(ex["target_values:"+v])<8: ex["target_values:"+v].append({"document_id":doc,"label":l,"value":v,**ctx(t,a,b)})
            left=t[a-1] if a else ""; right=t[b] if b<len(t) else ""
            for k,x in [("left_whitespace",bool(left and left.isspace())),("right_whitespace",bool(right and right.isspace())),("left_punctuation",bool(left and PUN.fullmatch(left))),("right_punctuation",bool(right and PUN.fullmatch(right))),("leading_whitespace",bool(v and v[0].isspace())),("trailing_whitespace",bool(v and v[-1].isspace())),("leading_punctuation",bool(v and PUN.fullmatch(v[0]))),("trailing_punctuation",bool(v and PUN.fullmatch(v[-1]))),("internal_newline","\n" in v),("leading_newline",v.startswith("\n")),("trailing_newline",v.endswith("\n"))]: c[k]+=x
            lp[l]["count"]+=1; lp[l]["chars"]+=len(v)
            if l=="NAME":
                lp[l]["honorific_included"]+=bool(HON.match(v))
                excluded=bool(a and HON_BEFORE.search(t[max(0,a-30):a]))
                lp[l]["honorific_excluded"]+=excluded
                if excluded and len(ex["honorific_excluded"])<12: ex["honorific_excluded"].append({"document_id":doc,"label":l,**ctx(t,a,b)})
                if HON.match(v) and len(ex["honorific_included"])<8: ex["honorific_included"].append({"document_id":doc,"label":l,**ctx(t,a,b)})
            if l=="USERNAME": lp[l]["numeric_or_reference_like"]+=bool(NUM.search(v))
            if (v[0].isspace() or v[-1].isspace() or PUN.fullmatch(v[0]) or PUN.fullmatch(v[-1])) and len(ex["edge_spans"])<12: ex["edge_spans"].append({"document_id":doc,"label":l,**ctx(t,a,b)})
            if (v[0].isspace() or v[-1].isspace()) and len(ex["whitespace_edge_spans"])<16: ex["whitespace_edge_spans"].append({"document_id":doc,"label":l,**ctx(t,a,b)})
            if "\n" in v and len(ex["multiline_spans"])<12: ex["multiline_spans"].append({"document_id":doc,"label":l,**ctx(t,a,b)})
            r=reg(t,a,b); rc[l][r]+=1; cd[l+":channel"][m["channel"]]+=1; cd[l+":domain"][m["domain"]]+=1; cd[l+":region"][r]+=1
            if l in {"NAME","USERNAME","JOB_TITLE"}:
                key=l+":contexts"
                if len(ex[key])<20: ex[key].append({"document_id":doc,"label":l,"metadata":{"channel":m["channel"],"domain":m["domain"]},**ctx(t,a,b,35)})
                cd[key]["preceding:"+re.sub(r"\s+"," ",t[max(0,a-35):a]).strip()[-24:]]+=1
                cd[key]["following:"+re.sub(r"\s+"," ",t[b:min(len(t),b+35)]).strip()[:24]]+=1
            cross[v][l]+=1
        if len(seen)<len(spans): rep["annotated_repeated_values"]+=len(spans)-len(seen); rep["documents_with_repeated_value"]+=1
        for v,occ in vals.items():
            labelled={(a,b) for a,b,_ in occ}; starts=[z.start() for z in re.finditer(re.escape(v),t)]; ul=[s for s in starts if (s,s+len(v)) not in labelled]
            if ul:
                rep["unlabelled_exact_occurrences"]+=len(ul); unlabelled_docs.add(doc)
                for s in ul:
                    if any(s < b and s+len(v) > a for a,b,_ in gold[doc]): rep["overlapping_unlabelled_occurrences"]+=1
                    else: rep["free_unlabelled_occurrences"]+=1
                if len(ex["unlabelled_repeats"])<12: ex["unlabelled_repeats"].append({"document_id":doc,"value":v,"labelled":[{"start":a,"end":b,"label":l} for a,b,l in occ],"unlabelled_offsets":ul,"text":t})
    rep["documents_with_unlabelled_repeat"]=len(unlabelled_docs)
    cross2=[{"value":v,"labels":dict(c),"total":sum(c.values())} for v,c in cross.items() if len(c)>1]; cross2.sort(key=lambda x:(-x["total"],x["value"]))
    minority=sum(sum(c.values())-max(c.values()) for c in cross.values() if len(c)>1)
    cross_examples={}
    for item in cross2[:3]:
        v=item["value"]; rows=[]; sampled_labels=collections.Counter()
        for doc,t in texts.items():
            for a,b,l in gold[doc]:
                if t[a:b]==v and sampled_labels[l]<3:
                    rows.append({"document_id":doc,"label":l,**ctx(t,a,b,30)})
                    sampled_labels[l]+=1
        cross_examples[v]=rows
    tops={k:[{"context":z,"count":n} for z,n in c.most_common(15)] for k,c in cd.items() if k.endswith(":contexts")}
    report={"scope":"train gold corpus only; no weak labels; no model training","denominators":totals,"definitions":{"left/right_whitespace":"immediately adjacent character exists and str.isspace() is true","left/right_punctuation":"immediately adjacent character exists and matches [^\\\\w\\\\s]","leading/trailing_whitespace":"first/last character inside the span is whitespace","leading/trailing_punctuation":"first/last character inside the span matches [^\\\\w\\\\s]","internal_newline":"annotation text contains newline; leading/trailing reported separately","honorific_included":"NAME begins with Mr/Mrs/Ms/Miss/Dr/Prof/Rev/Sir/Madam, optional period, case-insensitive","honorific_excluded":"NAME span is immediately preceded within 12 chars by the honorific pattern; proximity is evidence only","numeric_or_reference_like":"USERNAME contains digit, #, @, slash, hyphen, or ref/id/user/acct/ticket/case token","regions":"deterministic header/opening/body/paragraph-boundary/signature_or_tail heuristic in source","unlabelled_repeat":"same exact value occurs at an offset not present in gold spans; overlap is reported separately"},"boundary_by_label":{k:dict(v) for k,v in bd.items()},"label_profile":{k:dict(v) for k,v in lp.items()},"label_by_channel_domain_region":{k:dict(v) for k,v in cd.items() if not k.endswith(":contexts")},"region_by_label":{k:dict(v) for k,v in rc.items()},"repetition":rep,"cross_label_distinct_values":len(cross2),"cross_label_values_top":cross2[:100],"top_context_fragments":tops,"limitations":["Context fragments are descriptive, not causal evidence.","Repeated values are exact and case-sensitive.","Unlabelled repeats may reflect omission/context policy, not annotation error.","Regions are heuristics, not ground truth.","Metadata operational fields are not predictive evidence."]}
    report["cross_label_minority_assignments"]=minority
    report["cross_label_examples_top3"]=cross_examples
    report["definitions"]["honorific_excluded"]="A complete audited honorific ends the preceding 30-character window, followed only by whitespace before the NAME span; not a correctness claim."
    report["definitions"]["left/right_punctuation"]="Adjacent character is neither a Unicode word character nor whitespace."
    report["definitions"]["leading/trailing_punctuation"]="First/last character inside the span is neither a Unicode word character nor whitespace."
    report["definitions"]["unlabelled_repeat"]="Raw case-sensitive substring occurrence of a value labeled elsewhere in the same document, at different bounds; overlaps and free occurrences are separate. No word-boundary filtering."
    report["definitions"]["cross_label_minority_assignments"]="For each exact value, total label assignments minus its largest label count, summed across values. Descriptive disagreement, not an annotation error rate."
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"annotations.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    (OUT/"annotations_examples.json").write_text(json.dumps(dict(ex),indent=2,ensure_ascii=False),encoding="utf-8")
    md=["# Annotation convention audit and label-context profile","","Scope: train gold corpus only; reconstructed text and exact character offsets.","",f"Documents: {totals['documents']:,}; annotated documents: {totals['annotated_documents']:,}; annotations: {totals['annotations']:,}.",""]
    for l in labels:
        n=lp[l]["count"]; c=bd[l]; md.append(f"- **{l}**: {n:,} spans; left/right whitespace {pct(c['left_whitespace'],n):.2f}%/{pct(c['right_whitespace'],n):.2f}%; internal newline {pct(c['internal_newline'],n):.2f}%; punctuation adjacency {pct(c['left_punctuation'],n):.2f}%/{pct(c['right_punctuation'],n):.2f}%.")
    n=lp["NAME"]["count"]; md += ["",f"- **NAME honorifics**: {lp['NAME']['honorific_included']:,}/{n:,} ({pct(lp['NAME']['honorific_included'],n):.2f}%) begin with the audited pattern; {lp['NAME']['honorific_excluded']:,} NAME spans have a nearby preceding honorific pattern. These are boundary observations, not correctness claims.",f"- **USERNAME numeric/reference-like**: {lp['USERNAME']['numeric_or_reference_like']:,}/{lp['USERNAME']['count']:,} ({pct(lp['USERNAME']['numeric_or_reference_like'],lp['USERNAME']['count']):.2f}%) match the explicit format heuristic.",f"- **Exact repeats**: {rep['documents_with_repeated_value']:,} documents contain repeated annotated values; {rep['documents_with_unlabelled_repeat']:,} have raw exact matches at other offsets, but all {rep['overlapping_unlabelled_occurrences']:,} such matches overlap a larger gold span and {rep['free_unlabelled_occurrences']:,} are free. This scan provides no evidence of a free unlabelled repeat.",f"- **Cross-label exact values**: {len(cross2):,} distinct values occur with more than one label; {minority:,} assignments are minority-label occurrences among those values. Inspect cross_label_examples_top3 before using exact-value dictionaries.","","Full aggregates are in annotations.json; representative IDs/offset contexts are in annotations_examples.json. Structural regions and context fragments are descriptive heuristics."]
    (OUT/"annotations.md").write_text("\n".join(md)+"\n",encoding="utf-8")
    print(json.dumps({"documents":len(texts),"annotations":totals["annotations"],"cross_label_values":len(cross2),"repetition":rep,"output":str(OUT)},indent=2))
if __name__=="__main__": main()
