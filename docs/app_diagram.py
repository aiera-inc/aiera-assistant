from diagrams.onprem.client import Client, User
from diagrams.custom import Custom
from diagrams import Diagram, Cluster, Edge
from diagrams.generic.storage import Storage

with Diagram("Aiera Earnings Assistant", show=False, filename="docs/assistant", direction="TB"):
    user = User("User")
    client = Custom("Browser", "./assistant_snapshot.png")

    user >> client

    aiera = Custom("Aiera", "./aiera-icon-logo-circle.png")
    streamlit = Custom("Streamlit", "./streamlit.png")

    with Cluster("OpenAI"):
        openai = Custom("Assistant API", "./openai_logo.png")
        filestore = Storage("filestore")

    streamlit >>  Edge(label="serve app") >> client
    #streamlit >> Edge(label="start thread") >> openai
    openai >> Edge(label="process messages") << streamlit 
    streamlit >> Edge(label="get transcripts") << aiera

    streamlit >> Edge(label="upload transcript files") >> filestore
