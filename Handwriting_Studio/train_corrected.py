# Reproducible IAM / TrOCR fine-tuning; full data export and measured metrics.
import csv, hashlib, json, random, time, shutil, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from jiwer import cer, wer

ROOT=Path('/content/ocr2_results')
ROOT.mkdir(exist_ok=True)
REPORT=ROOT/'reports'; REPORT.mkdir(exist_ok=True)
SEED=42; EPOCHS=3; BATCH=8; LR=1e-5
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(2)
assert torch.cuda.is_available()
DEVICE='cuda'
def save(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2),encoding='utf-8')
def prep(im):
    im=ImageOps.exif_transpose(im).convert('RGBA')
    bg=Image.new('RGBA',im.size,'white'); bg.alpha_composite(im)
    return ImageOps.autocontrast(bg.convert('L'),cutoff=0.2).convert('RGB')
def score(ref,pred):
    ref=[' '.join(s.split()) for s in ref]; pred=[' '.join(s.split()) for s in pred]
    return {'cer':cer(ref,pred),'wer':wer(ref,pred),'exact_line_accuracy':sum(a==b for a,b in zip(ref,pred))/len(ref),'samples':len(ref)}
api=HfApi(); DATASET='Teklia/IAM-line'; BASE='microsoft/trocr-small-handwritten'
ds_rev=api.dataset_info(DATASET).sha; model_rev=api.model_info(BASE).sha
save(REPORT/'run_config.json',{'dataset':DATASET,'dataset_revision':ds_rev,'base_model':BASE,'model_revision':model_rev,'seed':SEED,'epochs':EPOCHS,'batch_size':BATCH,'learning_rate':LR,'encoder_frozen':True,'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'precision':'float32','train_size':'all','validation_size':256,'test_size':512})
print('Downloading IAM and TrOCR...',flush=True)
source=load_dataset(DATASET,revision=ds_rev)
processor=TrOCRProcessor.from_pretrained(BASE,revision=model_rev,use_fast=False)
model=VisionEncoderDecoderModel.from_pretrained(BASE,revision=model_rev,use_safetensors=False).to(DEVICE)
model.config.decoder_start_token_id=model.generation_config.decoder_start_token_id
model.config.pad_token_id=processor.tokenizer.pad_token_id
model.config.eos_token_id=processor.tokenizer.sep_token_id
model.generation_config.decoder_start_token_id=model.config.decoder_start_token_id
model.generation_config.pad_token_id=model.config.pad_token_id
model.generation_config.eos_token_id=model.config.eos_token_id
rows={}; manifest=[]; seen=set(); duplicates=[]
for split,limit in [('train',0),('validation',256),('test',512)]:
    indexes=list(range(len(source[split]))); random.Random(SEED).shuffle(indexes)
    rows[split]=[]
    for idx in indexes:
        if limit and len(rows[split])>=limit: break
        ex=source[split][idx]; im=ex['image'].convert('RGB')
        digest=hashlib.sha256(im.tobytes()+str(im.size).encode()).hexdigest()
        if digest in seen:
            duplicates.append({'split':split,'index':idx,'hash':digest}); continue
        seen.add(digest)
        row={'split':split,'source_index':idx,'text':ex['text'],'pixel_sha256':digest}
        for kind,img in [('raw',im),('preprocessed',prep(im))]:
            path=ROOT/'data'/kind/split/f'{idx:05}.png'; path.parent.mkdir(parents=True,exist_ok=True); img.save(path,compress_level=1)
            row[kind]=str(path.relative_to(ROOT))
        rows[split].append(row); manifest.append(row)
    print('Exported',split,len(rows[split]),flush=True)
with (ROOT/'data/manifest.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=manifest[0].keys()); w.writeheader(); w.writerows(manifest)
save(REPORT/'data_provenance.json',{'dataset':DATASET,'revision':ds_rev,'source_counts':{s:len(d) for s,d in source.items()},'used_counts':{s:len(d) for s,d in rows.items()},'excluded_exact_duplicates':duplicates,'all_new_training_images_included':True,'historical_pretraining_corpus_included':False,'note':'Source splits retained. Mirror lacks writer identifiers. Base model previously trained on IAM; not an independent unseen-corpus benchmark.'})
del source
# Persist sources inside the result package.
shutil.copy('/content/train_ocr.py',ROOT/'train_ocr.py')
if Path('/content/ocr_page.py').exists(): shutil.copy('/content/ocr_page.py',ROOT/'ocr_page.py')
(ROOT/'DATA_SOURCES.md').write_text('IAM: https://huggingface.co/datasets/Teklia/IAM-line\nOriginal terms: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database\nIAM is for non-commercial research; register and cite its authors as requested. Mirror images have already been resized to height 128. Raw PNGs are decoded mirror exports, not original archive bytes.\nBase model: https://huggingface.co/microsoft/trocr-small-handwritten\nAll images used in this new run are included. The complete historical pretraining corpus is not included.\nTrOCR recognizes lines. Page segmentation is a heuristic for clean single-column pages; no real-page accuracy is claimed.\n',encoding='utf-8')
(ROOT/'requirements.txt').write_text('torch>=2.6\ntransformers==4.57.1\ndatasets==4.3.0\njiwer==4.0.0\nopencv-python-headless==4.12.0.88\nsentencepiece==0.2.2\nPillow\n',encoding='utf-8')
def batch(part):
    px=processor(images=[Image.open(ROOT/r['preprocessed']).convert('RGB') for r in part],return_tensors='pt').pixel_values.to(DEVICE)
    tok=processor.tokenizer([r['text'] for r in part],padding=True,return_tensors='pt')
    if tok.input_ids.shape[1]>128: raise ValueError('Label exceeds 128 tokens')
    lab=tok.input_ids; lab[tok.attention_mask==0]=-100
    return px,lab.to(DEVICE)
@torch.inference_mode()
def evaluate(split,name):
    model.eval(); refs=[]; preds=[]; loss_total=0; n=0; started=time.time()
    for i in range(0,len(rows[split]),BATCH):
        part=rows[split][i:i+BATCH]; px,lab=batch(part)
        loss_total+=model(pixel_values=px,labels=lab).loss.item()*len(part); n+=len(part)
        ids=model.generate(px,max_new_tokens=128,num_beams=1)
        preds.extend(processor.batch_decode(ids,skip_special_tokens=True)); refs.extend(r['text'] for r in part)
        if i%(BATCH*16)==0: print(name,n,'/',len(rows[split]),'elapsed',round(time.time()-started),'s',flush=True)
    result=score(refs,preds); result['loss']=loss_total/n
    with (REPORT/f'{name}_predictions.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['source_index','reference','prediction']); w.writerows((r['source_index'],a,b) for r,a,b in zip(rows[split],refs,preds))
    save(REPORT/f'{name}_metrics.json',result); print(name,result,flush=True); return result
started=time.time(); baseline=evaluate('test','baseline_test')
for p in model.encoder.parameters(): p.requires_grad=False
optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=LR,weight_decay=.01)
history=[]; best=float('inf'); best_epoch=None
for epoch in range(1,EPOCHS+1):
    model.train(); model.encoder.eval(); shuffled=list(rows['train']); random.shuffle(shuffled); total=0; n=0
    for i in range(0,len(shuffled),BATCH):
        part=shuffled[i:i+BATCH]; px,lab=batch(part); optimizer.zero_grad(set_to_none=True)
        loss=model(pixel_values=px,labels=lab).loss
        if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step()
        total+=loss.item()*len(part); n+=len(part)
        if i%(BATCH*32)==0: print(f'Epoch {epoch}/{EPOCHS}: {n}/{len(shuffled)} lines, loss={total/n:.4f}, elapsed={time.time()-started:.0f}s',flush=True)
    val=evaluate('validation',f'epoch_{epoch}_validation'); history.append({'epoch':epoch,'train_loss':total/n,'validation':val})
    if val['cer']<best:
        best=val['cer']; best_epoch=epoch; model.save_pretrained(ROOT/'model'); processor.save_pretrained(ROOT/'model')
    save(REPORT/'history.json',history)
    print('Saved best checkpoint; epoch',best_epoch,flush=True)
del optimizer,model; torch.cuda.empty_cache()
model=VisionEncoderDecoderModel.from_pretrained(ROOT/'model').to(DEVICE)
final=evaluate('test','finetuned_test')
summary={'status':'completed','baseline_test':baseline,'finetuned_test':final,'selected_epoch':best_epoch,'selection_metric':'validation CER','history':history,'elapsed_seconds':time.time()-started,'real_page_metrics':None,'limitations':['Line metrics only, not real-page accuracy.','Base checkpoint already trained on IAM.','Heuristic page segmentation supports clean single-column pages.']}
save(REPORT/'metrics.json',summary)
(ROOT/'README.md').write_text('# English handwritten-page OCR\n\nTraining completed on a T4. See reports/metrics.json for measured results and data/manifest.csv for every image and label used. CER and WER are edit rates (lower is better); case and punctuation count, whitespace is normalized. Exact-line accuracy counts perfectly transcribed lines. Validation CER selects the checkpoint.\n\nUse: python ocr_page.py page.jpg --model model --out page_output\n\nReview page_output/transcription.txt and the line crops. Page segmentation supports clean, upright, single-column pages only; real-page accuracy has not been measured. Base model was previously trained on IAM. See DATA_SOURCES.md for research-use terms.\n',encoding='utf-8')
print('TRAINING COMPLETE',json.dumps(summary,indent=2),flush=True)
archive=shutil.make_archive('/content/ocr2_complete','zip',ROOT)
print('ALL RESULTS ARCHIVE:',archive,flush=True)
