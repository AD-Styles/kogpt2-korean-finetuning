import gradio as gr
import torch
import os
from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM

# 1. 모델 경로 유연화: 로컬 폴더가 없으면 현재 경로(HF Spaces 환경)를 참조
MODEL_DIR = "./kogpt2-korean-finetuned"
model_path = MODEL_DIR if os.path.exists(MODEL_DIR) else "."

# 2. 토크나이저 및 모델 로드
# 모델 파일(pytorch_model.bin 등)이 최상위 경로에 있어야 정상 작동합니다.
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2", 
    bos_token='</s>', eos_token='</s>', unk_token='<unk>', 
    pad_token='<pad>', mask_token='<mask()'
)
model = AutoModelForCausalLM.from_pretrained(model_path)

# 3. 디바이스 설정 및 추론 모드 전환
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

def generate_review(prompt):
    if not prompt: return "프롬프트를 입력해주세요!"
        
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=128,
            do_sample=True,
            temperature=0.8,
            top_p=0.92,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# 4. Gradio 인터페이스 레이아웃
demo = gr.Interface(
    fn=generate_review,
    inputs=gr.Textbox(lines=2, placeholder="영화 리뷰의 첫 문장을 입력하세요...", label="프롬프트"),
    outputs=gr.Textbox(label="생성된 리뷰"),
    title="🎬 Ko-GPT2 영화 리뷰 생성기",
    description="한국어 영화 리뷰 데이터셋(NSMC)으로 파인튜닝된 KoGPT2 모델입니다.",
    examples=["오늘 본 영화는", "이 영화의 결말은", "배우들의 연기가"]
)

if __name__ == "__main__":
    demo.launch()
