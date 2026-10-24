"""Ink & Type: a local/Colab interface for the delivered OCR checkpoint."""
import csv, json, os, tempfile, time, threading
from functools import lru_cache
from pathlib import Path
import gradio as gr
import torch
from PIL import Image, ImageDraw
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ocr_page import prep, segment

ROOT=Path(__file__).resolve().parent
MODEL=Path(os.environ.get('OCR_MODEL_PATH',str(ROOT/('base_model' if (ROOT/'base_model/config.json').exists() else 'model'))))
RUNS=Path(tempfile.gettempdir())/'ink_type_outputs'; RUNS.mkdir(exist_ok=True)
DEVICE='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
LOCK=threading.Lock()

@lru_cache(maxsize=1)
def engine():
    if not (MODEL/'config.json').exists():
        raise RuntimeError('Model files are missing. Place the trained model folder beside app.py.')
    p=TrOCRProcessor.from_pretrained(MODEL,use_fast=False,local_files_only=True)
    m=VisionEncoderDecoderModel.from_pretrained(MODEL,local_files_only=True).to(DEVICE).eval()
    return p,m

def empty(): return '',None,[], 'Upload a page or choose a sample to begin.',None

def recognize(image,mode,progress=gr.Progress()):
    if image is None: return '',None,[],'Choose an image first.',None
    if image.width*image.height>24_000_000: return '',None,[],'This image is too large. Resize it below 24 megapixels and try again.',None
    try:
        started=time.perf_counter(); progress(.03,desc='Preparing your page')
        if mode=='Single text line':
            page=prep(image); boxes=[[0,0,page.width,page.height]]
        else: page,boxes,_=segment(image)
        if not boxes: return '',page,[],'No text lines found. Try a brighter, closer image with dark writing.',None
        if len(boxes)>100: return '',page,[],'Too many possible text lines. Crop to one clean, single-column page.',None
        annotated=page.copy(); draw=ImageDraw.Draw(annotated); crops=[]; text=[]
        with LOCK,torch.inference_mode():
            p,m=engine()
            for i,box in enumerate(boxes):
                progress((i+.1)/len(boxes),desc=f'Reading line {i+1} of {len(boxes)}')
                crop=page.crop(box)
                px=p(images=prep(crop),return_tensors='pt').pixel_values.to(DEVICE)
                ids=m.generate(px,max_new_tokens=128,num_beams=1)
                line=p.batch_decode(ids,skip_special_tokens=True)[0]
                text.append(line);crops.append((crop,f'Line {i+1}: {line}'))
                draw.rectangle(box,outline='#245ae5',width=max(2,page.width//700))
                draw.text((box[0]+3,box[1]+2),str(i+1),fill='#163d9c')
        result='\n'.join(text); path=save_text(result)
        return result,annotated,crops,f'{len(boxes)} line(s) read in {time.perf_counter()-started:.1f}s. Review and correct the text before using it.',path
    except Exception as exc:
        return '',None,[],f'Could not read this image: {exc}',None

def save_text(text):
    if not text or not text.strip(): return None
    folder=Path(tempfile.mkdtemp(prefix='transcription_',dir=RUNS))
    path=folder/'transcription.txt'; path.write_text(text,encoding='utf-8'); return str(path)

using_base=MODEL.name=='base_model'
metric_path=ROOT/'reports/corrected_metrics.json' if using_base else ROOT/'reports/metrics.json'
metrics=json.loads(metric_path.read_text()) if metric_path.exists() else {}
result=metrics.get('baseline_test' if using_base else 'finetuned_test',{})
metric_rows=[[name,f'{100*result[key]:.2f}%' if key in result else 'Not measured'] for name,key in [('Character error rate','cer'),('Word error rate','wer'),('Perfectly transcribed lines','exact_line_accuracy')]]
examples=[]
manifest=ROOT/'data/manifest.csv'
if manifest.exists():
    with manifest.open(encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['split']=='test':
                examples.append([str(ROOT/row['raw']),'Single text line'])
                if len(examples)==3:break
assembled=ROOT/'assembled_page_demo/input.png'
if assembled.exists(): examples.insert(0,[str(assembled),'Full page'])

CSS='''
body .gradio-container{box-sizing:border-box!important;width:100vw!important;min-width:0!important;}
body .gradio-container main{width:100%!important;max-width:100%!important;min-width:0!important;box-sizing:border-box!important;}

.gradio-container {max-width:1200px!important; margin:auto!important; background:#f5f8fc!important; color:#152a43!important;}
#masthead {border-bottom:1px solid #cbd5e2;padding:24px 2px 22px;margin-bottom:16px;}
#masthead .wordmark{font-size:15px;font-weight:700;color:#245ae5;letter-spacing:.04em;}
#masthead h1{font-family:Georgia,serif!important;font-size:42px!important;line-height:1.12;margin:12px 0;color:#152a43!important;}
#masthead p{font-size:16px;color:#53657c;max-width:620px;line-height:1.6;}
#notice{background:#fff7e7;border-left:3px solid #b9780a;border-radius:4px;padding:12px 16px;font-size:14px;color:#59410e;}
#transcript textarea{font-family:Georgia,serif!important;font-size:19px!important;line-height:1.75!important;background:#fff!important;}
#read-button{background:#245ae5!important;border:none!important;color:white!important;min-height:48px;font-size:16px;}
#status{min-height:42px;color:#53657c;font-size:14px;}
#footnote{border-top:1px solid #cbd5e2;padding-top:14px;margin-top:18px;color:#53657c;font-size:13px;}
button:focus-visible,input:focus-visible,textarea:focus-visible{outline:3px solid #95b1ff!important;outline-offset:3px;}
@media(max-width:700px){#masthead h1{font-size:30px!important;}#masthead{padding-top:14px;} .gradio-container{padding:12px!important;width:100%!important;max-width:100%!important;min-width:0!important;box-sizing:border-box!important;}}
'''
theme=gr.themes.Base(primary_hue='blue',neutral_hue='slate').set(body_background_fill_dark='#f5f8fc',body_text_color_dark='#152a43',block_background_fill_dark='#ffffff',block_label_text_color_dark='#53657c',block_label_background_fill_dark='#f1f5fb',block_title_text_color_dark='#152a43',input_background_fill_dark='#ffffff',table_text_color_dark='#152a43',table_even_background_fill_dark='#ffffff',table_odd_background_fill_dark='#f1f5fb')
with gr.Blocks(title='Ink & Type — Handwriting reader',css=CSS,theme=theme,analytics_enabled=False,delete_cache=(3600,86400)) as demo:
    gr.HTML('<header id="masthead"><div class="wordmark">Ink & Type</div><h1>Your handwriting,<br>ready to edit.</h1><p>Bring a handwritten page into a text workspace. Read, compare, and keep the words that matter.</p></header>')
    gr.HTML(f'<div id="notice"><strong>Review your transcription.</strong> Measured line-test character error rate: {100*result.get("cer",0):.2f}%. Full-page accuracy has not been measured.</div>')
    with gr.Tab('Read handwriting'):
        with gr.Row(equal_height=True):
            with gr.Column(scale=1,min_width=300):
                image=gr.Image(label='Your handwritten page',type='pil',sources=['upload','clipboard'],height=380)
                mode=gr.Radio(['Full page','Single text line'],value='Full page',label='What are you uploading?')
                read=gr.Button('Read handwriting',variant='primary',elem_id='read-button')
                gr.Markdown('Use a clear **English, single-column** page. PNG, JPG or WebP. Keep the page upright and evenly lit.')
            with gr.Column(scale=1,min_width=300):
                text=gr.Textbox(label='Editable transcription',placeholder='Your text will appear here. You can correct it before downloading.',lines=14,interactive=True,show_copy_button=True,elem_id='transcript')
                status=gr.Markdown('Upload a page or choose a sample to begin.',elem_id='status')
                with gr.Row():
                    export=gr.Button('Save edited text')
                    clear=gr.Button('Start again')
                download=gr.File(label='Download your text',interactive=False)
        if examples:
            gr.Examples(examples=examples,inputs=[image,mode],label='Try a sample — test lines and one assembled page',examples_per_page=4)
        with gr.Accordion('Check what the model read',open=False):
            gr.Markdown('Blue boxes show the detected lines. Missing or merged boxes can cause missing text. Crop the image or switch to a single line if needed.')
            annotated=gr.Image(label='Detected lines',interactive=False,height=350)
            gallery=gr.Gallery(label='Line crops and readings',columns=2,height=300,object_fit='contain')
    with gr.Tab('Model results'):
        gr.Markdown('### Measured results\nActive model: **Original pretrained TrOCR-small**. Corrected fine-tuning regressed to 22.81% CER; the stronger unchanged checkpoint is used here.')
        gr.Dataframe(value=metric_rows,headers=['Metric','Measured value'],interactive=False)
        gr.Markdown(f"**Test lines:** {result.get('samples','Unavailable')} · **Selected epoch:** {'Not applicable' if using_base else metrics.get('selected_epoch','Unavailable')}\n\nTraining used 6,482 lines, with 256 validation lines and 512 test lines. All selected raw images, preprocessed images and labels are included in the results package.\n\nThe starting checkpoint was already trained on IAM. This is not an independent unseen-corpus benchmark. Multi-column pages, tables and overlapping writing are unsupported.")
        if metric_path.exists(): gr.File(value=str(metric_path),label='Download the measured metrics',interactive=False)
    gr.HTML('<footer id="footnote">English handwriting · Review before use · Your uploads are processed by the machine running this app.</footer>')
    outputs=[text,annotated,gallery,status,download]
    read.click(lambda:(' ',None,[],'Reading your image…',None),outputs=outputs,queue=False).then(recognize,inputs=[image,mode],outputs=outputs,concurrency_limit=1)
    export.click(save_text,inputs=text,outputs=download)
    clear.click(lambda:(None,'Full page',*empty()),outputs=[image,mode,*outputs],queue=False)
    image.change(empty,outputs=outputs,queue=False)
    mode.change(empty,outputs=outputs,queue=False)

def launch(port=7860):
    # Mounting avoids creating a public Gradio sharing link in Colab.
    from fastapi import FastAPI
    import uvicorn
    app=gr.mount_gradio_app(FastAPI(),demo.queue(),path='/')
    if 'google.colab' in __import__('sys').modules:
        threading.Thread(target=uvicorn.run,args=(app,),kwargs={'host':'0.0.0.0','port':port,'log_level':'warning'},daemon=True).start()
        from google.colab import output
        output.serve_kernel_port_as_iframe(port,height=1100)
    else: uvicorn.run(app,host='127.0.0.1',port=port)

if __name__=='__main__':
    engine()
    launch(int(os.environ.get('PORT','7860')))
