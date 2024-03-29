import nltk
import torch
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class Decoder:
    def decode(self, input):
        response = self.decoder_for_gpt(input)
        return response

    def decoder_for_gpt(self, input):
        chat = ChatOpenAI(model="gpt-3.5-turbo")
        xlt_msg = "I want you to act as a expert for summarization for Korean language. Here is an article below."
        template = ChatPromptTemplate.from_messages(
            [
                ("system", xlt_msg),
                ("human", "{question}"),
            ]
        )

        chain = template | chat
        response = chain.invoke({"question": input})
        return response

    def decoder_for_t5(self, input):
        model_dir = "lcw99/t5-large-korean-text-summary"
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

        max_input_length = 512 + 256
        inputs = ["summarize: " + input]
        input_ids = tokenizer(
            inputs, max_length=max_input_length, truncation=True, return_tensors="pt"
        )
        output = model.generate(
            **input_ids,
            do_sample=True,
            max_length=256,
            min_length=100,
            num_beams=8,
        )

        decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        response = nltk.sent_tokenize(decoded_output.strip())[0]
        return response
