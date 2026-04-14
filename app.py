import gradio as gr
import torch
from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM

# KoGPT2 모델 로드 (Hugging Face Hub의 공식 토크나이저 활용 - 가장 안정적)
# 수정됨: 모델 파일들이 최상위 경로에 있으므로 현재 폴더(".")를 지정합니다.
model_path = "."
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2", 
    bos_token='</s>', 
    eos_token='</s>', 
    unk_token='<unk>', 
    pad_token='<pad>', 
    mask_token='<mask>'
)
model = AutoModelForCausalLM.from_pretrained(model_path)

# GPU 사용 가능 시 이동
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

def generate_review(prompt):
    if not prompt:
        return "프롬프트를 입력해주세요!"
        
    # 입력 텍스트 토큰화
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # 텍스트 생성 파라미터 최적화
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
    
    # 생성된 토큰 ID를 텍스트로 복원
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Gradio 인터페이스 구성
demo = gr.Interface(
    fn=generate_review,
    inputs=gr.Textbox(lines=2, placeholder="영화 리뷰의 첫 문장을 적어보세요...", label="프롬프트"),
    outputs=gr.Textbox(label="생성된 리뷰"),
    title="🎬 Ko-GPT2 영화 리뷰 생성기",
    description="한국어 영화 리뷰 데이터셋(NSMC)으로 파인튜닝된 KoGPT2 모델입니다. 자연스러운 문장 생성을 위해 디코딩 전략이 최적화되어 있습니다.",
    examples=["오늘 본 영화는", "이 영화의 결말은", "배우들의 연기가"]
)

if __name__ == "__main__":
    demo.launch()
