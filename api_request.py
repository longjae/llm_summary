from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


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
