import PyPDF2
import chainlit as cl
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.callbacks.manager import CallbackManager
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


llm = ChatOllama(model='llama3', temperature=0.4, num_ctx=4096, Verbose=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

embeddings = OllamaEmbeddings(model='llama3')

prompt = PromptTemplate.from_template("""
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question} 

Context: {context} 

Answer: """)

@cl.on_chat_start
async def init():
    
    files = None
    pdf_text = ""
        
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a text file to begin!",
            accept=["application/pdf"],
            max_size_mb=200,
            max_files=10,
            timeout=180,
        ).send()


    text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2000,
                    chunk_overlap=200,
                    separators=["\n", ".", "?", "!"]
                    )


    for file in files:
        pdf = PyPDF2.PdfReader(file.path)
        for page in pdf.pages:
            pdf_text += page.extract_text()
            
    if len(files) == 1:
        msg = cl.Message(content=f"Processing `{files[0].name}`...", disable_feedback=True)
        await msg.send()
    
    else:
        msg = cl.Message(content=f"Processing files ...", disable_feedback=True)
        await msg.send()
        

    texts = text_splitter.split_text(pdf_text)
    print(len(texts))
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]
    
    vectordb = await cl.make_async(Chroma.from_texts)(
        texts, embedding=embeddings, persist_directory='./data', metadatas=metadatas)
    vectordb.persist()
    chain = RetrievalQA.from_chain_type(llm,
            chain_type="stuff",
            retriever=vectordb.as_retriever(),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,)
    
    
    msg.content = f"Processing done. You can now ask questions!"
    await msg.update()

    cl.user_session.set("metadatas", metadatas)
    cl.user_session.set("texts", texts)
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message):
    chain=cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
    stream_final_answer=True,
    answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached=True
    res=await chain.ainvoke(message.content, callbacks=[cb])
    answer=res["result"]
    answer=answer.replace(".",".  \n")
    sources=res["source_documents"]
    sources=str(str(sources)).replace("\n","  \n")
    
    if sources:
        answer+=f"\nSources: "+sources
    else:
        answer+=f"\nNo Sources found"

    await cl.Message(content=answer).send() 
    