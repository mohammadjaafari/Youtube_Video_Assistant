from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
_ = load_dotenv(find_dotenv()) # read local .env file

def create_db_from_youtube_video_url(video_url):
    
    loader = YoutubeLoader.from_youtube_url(video_url) #loading the video
    transcript = loader.load() 

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100) #splitting by 1000 chunks
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, OpenAIEmbeddings()) #faiss is a vector database we will use to handle
    return db                                           # the vector embedding using openai embedding


def get_response_from_query(db, query, k=4): #gpt-3.5-turbo can handle up to 4097 tokens. 
                                             #Setting the chunksize to 1000 and k to 4 maximizes
                                             #the number of tokens to analyze.
    

    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs]) #join the chunks after the vector db search

    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)

    # Template to use for the system message prompt
    template = """
        You are an assistant that aims to answer questions related to the context of youtube
        videos based on their transcript below:
        {docs}
        
        Only use relevant information from the transcript to answer the question.
        
        Keep your answer as summarized as possible without missing important detail.
        
        If you are not able to answer the question ask the user to enter a clearer question relevant
        to the video
        """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Human question prompt
    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt) #create the llm chain

    response = chain.run(question=query, docs=docs_page_content) #run the chain
    response = response.replace("\n", "")
    return response, docs

while(True):
    
    print("Enter video url | Type 'Exit' to exit ")
    video_url = input()
    if video_url.lower() == "exit":
        break
    

    db = create_db_from_youtube_video_url(video_url)

    print("\n What do you want to ask about the video? | Type 'Exit' to exit")
    

    query = input()

    
    if query.lower() == "exit":
        break
    response, docs = get_response_from_query(db, query)
    print("\n" + response)
    
    print("-----------------------------------------------------")