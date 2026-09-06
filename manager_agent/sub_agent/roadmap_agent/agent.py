import os
import random
from typing import Dict, Any, List, Literal, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Try importing Bedrock AgentCore & Playwright for Managed Browser Tool
try:
    from bedrock_agentcore.tools.browser_client import browser_session
    from playwright.async_api import async_playwright
except ImportError:
    browser_session = None
    async_playwright = None

# Google ADK imports
from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import ToolContext, FunctionTool

# Load environment variables
load_dotenv(".env")


# ----------------------------------------------------------------------
# 1. DEFINE SCHEMAS (Pydantic Models)
# ----------------------------------------------------------------------

class SkillPreferenceInput(BaseModel):
    """
    Structured extraction of a student's existing skills, experiences,
    and learning focuses to customize their path.
    """
    mastered_nodes: List[str] = Field(
        default_factory=list,
        description="List of node IDs (e.g. ['Node-1']) that the student is confident they have fully mastered."
    )
    prioritized_nodes: List[str] = Field(
        default_factory=list,
        description="List of node IDs (e.g. ['Node-3']) that the student wants to prioritize and learn first."
    )
    user_skills_text: Optional[str] = Field(
        default=None, \
        description="Free text describing what the student already knows or is interested in."
    )


# ----------------------------------------------------------------------
# 2. DEFINE ROADMAPS CATALOG AND STATE MACHINE TOOLS
# ----------------------------------------------------------------------

ROADMAPS_CATALOG = {
    "AI Engineer": [
        {
            "id": "Node-1",
            "title": "Programming & System Foundations",
            "description": "Python OOP, APIs integration (FastAPI/Flask), SQL queries, and Vector DBs (Chroma/Pinecone).",
            "status": "pending",
            "priority": "normal",
            "topics": ["Python OOP basics", "FastAPI / Flask REST APIs", "SQL Database queries",
                       "Vector Databases (Chroma/Pinecone)"],
            "resources": {
                "beginner": "HuggingFace NLP Course Unit 1 & Python basics.",
                "intermediate": "FastAPI official documentation & SQL querying course.",
                "master": "Designing high-throughput microservices in Python."
            }
        },
        {
            "id": "Node-2",
            "title": "Machine Learning & Deep Learning Basics",
            "description": "Linear regression, multi-class classification, neural networks from scratch using PyTorch.",
            "status": "pending",
            "priority": "normal",
            "topics": ["Linear & Logistic Regression", "Multi-class Classification models",
                       "Neural Networks from scratch", "PyTorch fundamentals"],
            "resources": {
                "beginner": "Coursera's Machine Learning Specialization by Andrew Ng.",
                "intermediate": "Deep Learning with PyTorch: A 60 Minute Blitz.",
                "master": "Designing Machine Learning Systems by Chip Huyen."
            }
        },
        {
            "id": "Node-3",
            "title": "Generative AI & Semantic Search (RAG)",
            "description": "Prompt engineering patterns, Retrieval-Augmented Generation, document parsing, and semantic search.",
            "status": "pending",
            "priority": "normal",
            "topics": ["Prompt Engineering patterns", "Retrieval-Augmented Generation (RAG)",
                       "Document Chunking & Parsing", "Semantic Vector Search"],
            "resources": {
                "beginner": "DeepLearning.AI: Prompt Engineering for Developers.",
                "intermediate": "LangChain & LlamaIndex documentation for RAG applications.",
                "master": "Advanced RAG pipelines (hybrid search, re-ranking, and evaluations)."
            }
        },
        {
            "id": "Node-4",
            "title": "Agentic AI & Multi-Agent Orchestration",
            "description": "Autonomous agents, tool use, loop architectures, and orchestration frameworks (LangGraph/CrewAI/ADK).",
            "status": "pending",
            "priority": "normal",
            "topics": ["AI Agent fundamentals", "LangChain & LangGraph", "Multi-Agent Systems",
                       "Tool Use & Function Calling"],
            "resources": {
                "beginner": "Hacking with AI Agents: Basic workflows.",
                "intermediate": "Google Agent Development Kit (ADK) & LangGraph state guides.",
                "master": "Multi-agent orchestration & hierarchical supervisory system patterns."
            }
        },
        {
            "id": "Node-5",
            "title": "MLOps & Scalable Deployment",
            "description": "Docker containerization, CI/CD pipelines, model latency monitoring, and cloud deployment (AWS/GCP).",
            "status": "pending",
            "priority": "normal",
            "topics": ["Docker containerization", "CI/CD Deployment Pipelines", "Model Serving (Triton/FastAPI)",
                       "Kubernetes cluster orchestrations"],
            "resources": {
                "beginner": "Docker for Beginners (hands-on tutorial).",
                "intermediate": "Automating deployments with GitHub Actions & AWS ECR.",
                "master": "Scaling production model inference with Triton Server & Kubernetes."
            }
        }
    ],
    "Data Scientist": [
        {
            "id": "Node-1",
            "title": "Mathematics, Probability & Statistics",
            "description": "Descriptive statistics, probability distributions, hypothesis testing, and statistical experiments.",
            "status": "pending",
            "priority": "normal",
            "topics": ["Descriptive Statistics", "Probability Distributions", "Hypothesis Testing",
                       "Statistical Experiments & A/B testing"],
            "resources": {
                "beginner": "Khan Academy Statistics & Probability course.",
                "intermediate": "Practical Statistics for Data Scientists (O'Reilly).",
                "master": "The Elements of Statistical Learning (Hastie, Tibshirani)."
            }
        },
        {
            "id": "Node-2",
            "title": "Programming & Data Wrangling (SQL & Pandas)",
            "description": "Python data science libraries (Pandas, Numpy), SQL databases, structured queries, and data cleaning.",
            "status": "pending",
            "priority": "normal",
            "topics": ["Pandas Data Wrangling & Manipulation", "NumPy arrays", "SQL Database queries",
                       "Data Cleaning Techniques"],
            "resources": {
                "beginner": "Kaggle Pandas & Intro to SQL tutorials.",
                "intermediate": "Python for Data Analysis by Wes McKinney.",
                "master": "Advanced SQL queries, window functions, and query plan optimization."
            }
        },
        {
            "id": "Node-3",
            "title": "Machine Learning & Predictive Modeling",
            "description": "Supervised/unsupervised machine learning models (regression, classification, clustering, tree-based models).",
            "status": "pending",
            "priority": "normal",
            "topics": ["Supervised Learning (Regression/Classification)", "Unsupervised Clustering",
                       "Tree-based Models & Random Forest", "Feature Engineering"],
            "resources": {
                "beginner": "Kaggle Machine Learning course.",
                "intermediate": "Hands-On Machine Learning with Scikit-Learn (Aurélien Géron).",
                "master": "Feature Engineering for Machine Learning by Alice Zheng."
            }
        },
        {
            "id": "Node-4",
            "title": "Data Visualization & BI Storytelling",
            "description": "Data exploration (Matplotlib/Seaborn), Tableau/PowerBI, translating model metrics into business insights.",
            "status": "pending",
            "priority": "normal",
            "topics": ["Matplotlib & Seaborn plotting", "Tableau & PowerBI visualization",
                       "Business Intelligence Storytelling", "Executive Stakeholder Reporting"],
            "resources": {
                "beginner": "Data Visualization with Matplotlib & Seaborn.",
                "intermediate": "Storytelling with Data by Cole Nussbaumer Knaflic.",
                "master": "Building Executive Dashboards & Enterprise Business Intelligence strategies."
            }
        },
        {
            "id": "Node-5",
            "title": "Generative AI for Advanced Analytics",
            "description": "Using LLM APIs for automated sentiment analysis, feature extraction, and text summary reports.",
            "status": "pending",
            "priority": "normal",
            "topics": ["NLP Text Wrangling", "LLM API Sentiment Analysis", "Feature Extraction with LLMs",
                       "Automated Tabular parsing"],
            "resources": {
                "beginner": "Prompt Engineering for Business Analysts.",
                "intermediate": "Using HuggingFace models for automated tabular parsing.",
                "master": "Deploying custom analytical copilots inside enterprise DB architectures."
            }
        }
    ],
    "MLOps Engineer": [
        {
            "id": "Node-1",
            "title": "System Administration & Infrastructure (Linux)",
            "description": "Linux terminal (bash), shell scripting, SSH configurations, file systems, and network permissions.",
            "status": "pending",
            "priority": "normal",
            "topics": ["Linux Terminal Bash CLI", "Automated Shell Scripting", "SSH Configurations",
                       "Network Permissions & Security"],
            "resources": {
                "beginner": "Linux Journey (interactive guide) & basic bash scripting.",
                "intermediate": "The Linux Command Line by William Shotts.",
                "master": "Enterprise automated server provisioning using Ansible."
            }
        },
        {
            "id": "Node-2",
            "title": "Containerization & Orchestration",
            "description": "Docker container creation, multi-container configurations, microservices architecture, and Kubernetes.",
            "status": "pending",
            "priority": "normal", \
            "topics": ["Docker Containers creation", "Microservices Architecture", "Kubernetes Orchestration",
                       "GPU Cluster Scheduling"],
            "resources": {
                "beginner": "Docker Handbook (freeCodeCamp tutorial).",
                "intermediate": "Docker & Kubernetes: The Complete Guide (Udemy).",
                "master": "Kubernetes in Action by Marko Lukša."
            }
        },
        {
            "id": "Node-3",
            "title": "ML Pipelines & Version Control",
            "description": "Git pipelines, MLflow tracking, registry, and DVC (Data Version Control) for models and datasets.",
            "status": "pending",
            "priority": "normal", \
            "topics": ["Git Pipelines", "MLflow Registry & Tracking", "Data Version Control (DVC)",
                       "Data Pipelines & ETL"],
            "resources": {
                "beginner": "Git & GitHub Basics (interactive tutorial).",
                "intermediate": "MLflow tracking guide & registering models.",
                "master": "Designing large-scale data and model versioning infrastructures."
            }
        },
        {
            "id": "Node-4",
            "title": "High-Performance Serving & Serving Engines",
            "description": "Model serving using Triton Inference Server, TFServing, FastAPI integrations, and TensorRT model optimization.",
            "status": "pending",
            "priority": "normal", \
            "topics": ["Triton Inference Server configuration", "FastAPI integrations", "TensorRT model optimization",
                       "ONNX conversions"],
            "resources": {
                "beginner": "Serving simple models as REST endpoints.",
                "intermediate": "Model optimization with TensorRT & ONNX runtime conversion.",
                "master": "Triton Server configurations, dynamic batching, and GPU cluster scheduling."
            }
        },
        {
            "id": "Node-5",
            "title": "CI/CD & Monitoring Systems",
            "description": "GitHub Actions for automated deployment, drift detection, system throughput and model performance monitoring.",
            "status": "pending", \
            "priority": "normal", \
            "topics": ["Automated CI/CD with GitHub Actions", "Grafana & Prometheus metrics dashboards",
                       "Drift and Latency Detection", "Canary releases"],
            "resources": {
                "beginner": "GitHub Actions basic pipeline tutorials.",
                "intermediate": "Grafana & Prometheus dashboards for monitoring live server metrics.",
                "master": "Continuous deployment pipeline with shadow rollouts and canary releases."
            }
        }
    ],
    "NLP / LLM Specialist": [
        {
            "id": "Node-1",
            "title": "Linguistics & Text Preprocessing",
            "description": "Regular expressions, tokenization, stemming/lemmatization, parsing strings, Spacy and NLTK.",
            "status": "pending",
            "priority": "normal", \
            "topics": ["Regular Expressions (regex)", "Tokenization & Lemmatization", "Parsing strings",
                       "Spacy and NLTK"],
            "resources": {
                "beginner": "Introduction to NLP with NLTK tutorials.",
                "intermediate": "Spacy Advanced NLP Course & text wrangling strategies.",
                "master": "Designing custom parsers and syntax engines for unstructured communications."
            }
        },
        {
            "id": "Node-2",
            "title": "Deep Learning & Transformers",
            "description": "RNNs, LSTMs, Attention mechanism, Transformer architectures, and HuggingFace pre-trained models.",
            "status": "pending",
            "priority": "normal", \
            "topics": ["Recurrent Neural Networks (RNNs) & LSTMs", "Attention Mechanism", "Transformer Architectures",
                       "HuggingFace pre-trained models"],
            "resources": {
                "beginner": "RNNs & LSTMs deep dive (DeepLearning.AI).",
                "intermediate": "HuggingFace NLP Course (official tutorial).",
                "master": "Attention Is All You Need (original paper replication in PyTorch)."
            }
        },
        {
            "id": "Node-3",
            "title": "LLM Systems & Semantic Search (RAG)",
            "description": "Prompt Engineering, Retrieval-Augmented Generation, and Vector DB integration (Chroma/Milvus).",
            "status": "pending",
            "priority": "normal", \
            "topics": ["Prompt Patterns", "RAG Vector Search", "LlamaIndex & LangChain",
                       "Document Intelligence & Parsing"],
            "resources": {
                "beginner": "HuggingFace prompt engineering documentation.",
                "intermediate": "LlamaIndex & LangChain framework implementation.",
                "master": "Hybrid vector search, semantic chunking, and multi-turn conversational agents."
            }
        },
        {
            "id": "Node-4",
            "title": "Model Fine-tuning & Alignment",
            "description": "Supervised Fine-Tuning (SFT), PEFT (LoRA, QLoRA), and RLHF alignment strategies.",
            "status": "pending",
            "priority": "normal", \
            "topics": ["Supervised Fine-Tuning (SFT)", "Parameter-Efficient Fine-Tuning (PEFT/LoRA)",
                       "Direct Preference Optimization (DPO)", "RLHF alignment"],
            "resources": {
                "beginner": "Fine-tuning model weights using HuggingFace Trainer.",
                "intermediate": "LoRA & QLoRA fine-tuning in PyTorch (hands-on project).",
                "master": "DPO (Direct Preference Optimization) implementation."
            }
        },
        {
            "id": "Node-5",
            "title": "Enterprise Deployment & Guardrails",
            "description": "High-performance serving (vLLM, Ollama), Guardrails frameworks (NeMo Guardrails), and quantization techniques.",
            "status": "pending",
            "priority": "normal", \
            "topics": ["High-performance LLM serving (vLLM/Ollama)", "Guardrails Frameworks (NeMo Guardrails)",
                       "Quantization Techniques (AWQ/GPTQ)", "Model Evaluations"],
            "resources": {
                "beginner": "Local model serving with Ollama.",
                "intermediate": "vLLM deployment for multi-concurrency streaming REST endpoints.",
                "master": "Designing safety layers, prompt jailbreak guardrails, and model evaluations."
            }
        }
    ],
    "Computer Vision Engineer": [
        {
            "id": "Node-1",
            "title": "Image Processing & OpenCV Foundations",
            "description": "Pixel array manipulations, kernel filters, spatial coordinates, geometry, and image transforms.",
            "status": "pending",
            "priority": "normal", \
            "topics": ["Pixel array manipulation", "Kernel filters", "Spatial Coordinate Systems", "2D/3D Geometry"],
            "resources": {
                "beginner": "OpenCV basics in Python tutorial.",
                "intermediate": "Learning OpenCV 3 Computer Vision Book.",
                "master": "Writing parallelized custom GPU image operations."
            }
        },
        {
            "id": "Node-2",
            "title": "Deep Learning & CNNs",
            "description": "Convolutional Neural Networks, architecture design (ResNet, EfficientNet), image classification with PyTorch.",
            "status": "pending",
            "priority": "normal", \
            "topics": ["Convolutional Neural Networks (CNNs)", "ResNet/EfficientNet", "PyTorch Image Classification",
                       "Data Augmentation"],
            "resources": {
                "beginner": "PyTorch image classification quickstart.",
                "intermediate": "Deep Learning for Computer Vision (Coursera).",
                "master": "Neural Architecture Search (NAS) for mobile/embedded CV applications."
            }
        },
        {
            "id": "Node-3",
            "title": "Object Detection & Segmentation",
            "description": "YOLO algorithms, object tracking, semantic and instance segmentation (U-Net, Mask R-CNN).",
            "status": "pending",
            "priority": "normal", \
            "topics": ["YOLO Custom Object Detection", "Object Tracking", "Semantic Segmentation (U-Net)",
                       "Instance Segmentation (Mask R-CNN)"],
            "resources": {
                "beginner": "YOLOv8 custom object detector tutorial.",
                "intermediate": "Custom training pipelines for object detection & segmentation.",
                "master": "Segment Anything Model (SAM) fine-tuning."
            }
        },
        {
            "id": "Node-4",
            "title": "Edge CV & Robotic Integration",
            "description": "Inference optimization on embedded devices (TensorRT, OpenVINO), Camera pipelines, and robotic Locomotion SDKs.",
            "status": "pending",
            "priority": "normal", \
            "topics": ["Embedded CV Deployment (Raspberry Pi/Jetson)", "TensorRT optimization & Quantization",
                       "VLM Live Clients Integration", "Locomotion SDKs Interface"],
            "resources": {
                "beginner": "Deploying classification models on Raspberry Pi.",
                "intermediate": "TensorRT quantization (INT8 calibration) for embedded cameras.",
                "master": "NaVILA Client: Interfacing visual frames with Locomotion policies in simulated environments."
            }
        },
        {
            "id": "Node-5",
            "title": "Vision-Language Models & Generative Vision",
            "description": "Stable Diffusion, Image generation, VLMs, visual grounding, and multimodal embeddings.",
            "status": "pending",
            "priority": "normal", \
            "topics": ["CLIP Multimodal Embeddings", "Stable Diffusion pipeline fine-tuning", "Visual Grounding",
                       "Generative Vision Models"],
            "resources": {
                "beginner": "Multimodal embeddings with CLIP model.",
                "intermediate": "Stable Diffusion pipeline fine-tuning (LoRA).",
                "master": "Autoregressive VLM fine-tuning."
            }
        }
    ],
    "ML Research Scientist": [
        {
            "id": "Node-1",
            "title": "Advanced Mathematics & ML Theory",
            "description": "Linear algebra, multivariate calculus, optimization, and statistical learning theory proofs.",
            "status": "pending",
            "priority": "normal", \
            "topics": ["Advanced Linear Algebra", "Multivariate Calculus", "Mathematical Optimization Theory",
                       "Convex Optimization"],
            "resources": {
                "beginner": "MIT 18.06 Linear Algebra by Gilbert Strang.",
                "intermediate": "Mathematics for Machine Learning (Cambridge Book).",
                "master": "Convex Optimization by Stephen Boyd."
            }
        },
        {
            "id": "Node-2",
            "title": "Deep Neural Network Theory",
            "description": "Mathematical derivations of backpropagation, optimization algorithms convergence, generalization bounds.",
            "status": "pending",
            "priority": "normal", \
            "topics": ["Derivations of backpropagation", "Convergence of optimization algorithms",
                       "Generalization Bounds", "Statistical Learning Theory"],
            "resources": {
                "beginner": "Backpropagation mathematics (animated guides).",
                "intermediate": "Deep Learning textbook (Goodfellow, Bengio).",
                "master": "Generalization bounds and statistical learning theories."
            }
        },
        {
            "id": "Node-3",
            "title": "Literature Replication & JAX Programming",
            "description": "Replicating SOTA research papers (NeurIPS, ICML), and compiling high-performance tensor code in JAX.",
            "status": "pending",
            "priority": "normal", \
            "topics": ["Research Paper Replication", "JAX programming tutorials", "Autodiff and Pytrees",
                       "Custom backpropagation kernels"],
            "resources": {
                "beginner": "JAX tutorial: Autodiff and Pytrees.",
                "intermediate": "Replicating classical architectures (like ResNet/Attention) in JAX.",
                "master": "Writing custom compilers and backpropagation kernels in JAX."
            }
        },
        {
            "id": "Node-4",
            "title": "Generative Foundations & Scaling Laws",
            "description": "Diffusion mathematics (SDEs), Scaling laws of Transformers, attention theoretical complexity bounds.",
            "status": "pending",
            "priority": "normal", \
            "topics": ["Score-based Generative Models (Diffusion mathematics)", "Transformer Scaling Laws",
                       "Empirical Scaling Law Calculations", "Theoretical Complexity bounds"],
            "resources": {
                "beginner": "Transformers mathematical formulations.",
                "intermediate": "Score-based generative models tutorial.",
                "master": "Empirical Scaling Laws of transformers (Chinchilla calculations)."
            }
        },
        {
            "id": "Node-5",
            "title": "AI Safety & Alignment Research",
            "description": "Mechanistic interpretability of model weights, safety validation, formal alignment bounds.",
            "status": "pending",
            "priority": "normal", \
            "topics": ["Mechanistic Interpretability of model weights", "Safety Validation frameworks",
                       "RLHF mathematical alternatives", "DPO bounds"],
            "resources": {
                "beginner": "Intro to Mechanistic Interpretability by Neel Nanda.",
                "intermediate": "Formal verification of neural networks.",
                "master": "AI Safety research: RLHF mathematical alternatives & DPO bounds."
            }
        }
    ]
}


def reset_roadmap_session(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Resets the roadmap session variables in the persistent session state.
    """
    tool_context.state["learning_nodes"] = {}
    tool_context.state["roadmap_initialized"] = False
    tool_context.state["career_fit"] = ""
    print("\n[ROADMAP ENGINE] Roadmap session state reset.")
    return {"status": "success", "message": "Roadmap session has been completely reset."}


def initialize_student_roadmap(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Initializes the learning nodes list programmatically based on the primary recommendation
    saved in session state by the career consulting agent, keeping track of persistent state variables.
    """
    # 1. Retrieve the matched career path from consulting agent's state
    career_results = tool_context.state.get("career_fit_results", {})
    primary_recommendation = career_results.get("primary_recommendation", "AI Engineer")

    # Save to state
    tool_context.state["career_fit"] = primary_recommendation

    # 2. Fetch the standard nodes from catalog
    standard_nodes = ROADMAPS_CATALOG.get(primary_recommendation, ROADMAPS_CATALOG["AI Engineer"])

    # 3. Save as a dict for mutable, dynamic access by ID, ensuring completed_topics is initialized!
    learning_nodes = {}
    for node in standard_nodes:
        node_copy = dict(node)
        node_copy["completed_topics"] = list(node.get("completed_topics", []))
        node_copy["progress_percentage"] = float(node.get("progress_percentage", 0.0))
        learning_nodes[str(node["id"])] = node_copy

    tool_context.state["learning_nodes"] = learning_nodes
    tool_context.state["roadmap_initialized"] = True

    print(
        f"\n[ROADMAP ENGINE] Programmatically initialized default '{primary_recommendation}' roadmap with {len(standard_nodes)} nodes.")

    return {
        "status": "success",
        "career_path": primary_recommendation,
        "nodes": list(learning_nodes.values()),
        "instruction": "Display this checklist to the student as a beautiful numbered list with progress bars for each milestone."
    }


def rearrange_roadmap_by_preferences(user_preference_text: str, tool_context: ToolContext) -> Dict[str, Any]:
    """
    ADAPTIVE ALIGNMENT ENGINE:
    Analyzes student's free-text skills description or interests, and programmatically updates
    roadmap priorities or marks topics as completed. It then re-orders nodes (High Priority first,
    Normal next, Low next, Completed last) to keep the roadmap personalized and transformative.
    """
    if "learning_nodes" not in tool_context.state or not tool_context.state.get("roadmap_initialized", False):
        initialize_student_roadmap(tool_context)

    nodes = tool_context.state["learning_nodes"]
    text_lower = user_preference_text.lower()
    career_path = tool_context.state.get("career_fit", "AI Engineer")

    # Ensure completed_topics is initialized
    for node in nodes.values():
        if "completed_topics" not in node:
            node["completed_topics"] = []

    # Map topics to keywords for high-fidelity matching
    TOPIC_KEYWORDS_MAP = {
        # AI Engineer
        "python oop basics": ["python", "oop", "object-oriented", "class", "classes", "inheritance", "polymorphism"],
        "fastapi / flask rest apis": ["fastapi", "flask", "rest api", "apis", "api", "backend", "web services"],
        "sql database queries": ["sql", "database", "postgres", "queries", "querying", "mysql", "sqlite",
                                 "relational database"],
        "vector databases (chroma/pinecone)": ["vector database", "vector db", "chroma", "pinecone", "milvus",
                                               "embeddings", "qdrant", "vector store"],
        "linear & logistic regression": ["linear regression", "logistic regression", "regression", "curve fitting"],
        "multi-class classification models": ["multi-class", "classification", "classifier", "random forest",
                                              "decision tree"],
        "neural networks from scratch": ["neural networks from scratch", "neural network from scratch", "building nn",
                                         "backprop from scratch"],
        "pytorch fundamentals": ["pytorch", "tensor", "tensors", "torch"],
        "prompt engineering patterns": ["prompt engineering", "prompting", "few-shot", "zero-shot", "prompt patterns",
                                        "system prompts"],
        "retrieval-augmented generation (rag)": ["rag", "retrieval-augmented generation", "semantic search", "chunking",
                                                 "retrieval"],
        "document chunking & parsing": ["document chunking", "parsing", "chunking", "unstructured", "pdf parser"],
        "semantic vector search": ["semantic search", "vector search", "similarity search", "cosine similarity"],
        "ai agent fundamentals": ["agent fundamentals", "ai agent", "autonomous agent", "react loop", "agentic"],
        "langchain & langgraph": ["langchain", "langgraph", "agent orchestration", "graphs"],
        "multi-agent systems": ["multi-agent", "agent team", "crewai", "autogen", "cooperative agents"],
        "tool use & function calling": ["tool use", "function calling", "bind tools", "tool calling"],
        "docker containerization": ["docker", "container", "containerization", "dockerfile", "containers"],
        "ci/cd deployment pipelines": ["ci/cd", "continuous integration", "github actions", "pipelines", "pipeline"],
        "model serving (triton/fastapi)": ["model serving", "serving", "triton", "tfserving", "inference server"],
        "kubernetes cluster orchestrations": ["kubernetes", "k8s", "cluster", "helm", "orchestration"],

        # Data Scientist
        "descriptive statistics": ["descriptive statistics", "mean", "median", "variance", "std dev",
                                   "summary statistics"],
        "probability distributions": ["probability distributions", "probability", "distribution", "normal distribution",
                                      "bayes theorem"],
        "hypothesis testing": ["hypothesis testing", "p-value", "t-test", "anova", "z-test"],
        "statistical experiments & a/b testing": ["experiments", "a/b testing", "ab testing", "experimental design",
                                                  "significance"],
        "pandas data wrangling & manipulation": ["pandas", "dataframe", "data wrangling", "data manipulation",
                                                 "wrangling"],
        "numpy arrays": ["numpy", "arrays", "matrices", "numerical python"],
        "data cleaning techniques": ["data cleaning", "imputation", "outliers", "missing values", "clean data"],
        "supervised learning (regression/classification)": ["supervised", "regression", "classification", "svm", "knn",
                                                            "scikit-learn"],
        "unsupervised clustering": ["unsupervised", "clustering", "k-means", "pca", "dimensionality reduction"],
        "tree-based models & random forest": ["tree-based", "random forest", "decision tree", "xgboost",
                                              "gradient boosting"],
        "feature engineering": ["feature engineering", "feature selection", "scaling", "one-hot encoding"],
        "matplotlib & seaborn plotting": ["matplotlib", "seaborn", "plotting", "charts", "data visualization",
                                          "graphs"],
        "tableau & powerbi visualization": ["tableau", "powerbi", "power bi", "dashboards", "dashboard"],
        "business intelligence storytelling": ["storytelling", "business intelligence", "bi", "story", "insights"],
        "executive stakeholder reporting": ["executive", "stakeholder", "reporting", "presentation", "report"],
        "nlp text wrangling": ["text wrangling", "text processing", "regex", "nlp data"],
        "llm api sentiment analysis": ["sentiment analysis", "llm api", "sentiment", "text classification"],
        "feature extraction with llms": ["feature extraction", "text embeddings", "embeddings from text"],
        "automated tabular parsing": ["tabular parsing", "table extraction", "excel parsing",
                                      "structured data extraction"],

        # MLOps Engineer
        "linux terminal bash cli": ["linux", "terminal", "bash", "cli", "command line", "shell scripting", "shell"],
        "automated shell scripting": ["shell scripting", "bash script", "automation scripts", "cron"],
        "ssh configurations": ["ssh", "secure shell", "ssh keys", "remote login"],
        "network permissions & security": ["network", "permissions", "security", "firewall", "vpc", "iam"],
        "docker containers creation": ["docker", "container", "containers", "dockerfile"],
        "microservices architecture": ["microservices", "microservice", "soa", "decoupled"],
        "kubernetes orchestration": ["kubernetes", "k8s", "orchestration", "pods", "deployments"],
        "gpu cluster scheduling": ["gpu", "gpu cluster", "cuda", "scheduling", "slurm"],
        "git pipelines": ["git", "github", "gitlab", "version control", "pr", "commits"],
        "mlflow registry & tracking": ["mlflow", "model registry", "experiment tracking"],
        "data version control (dvc)": ["dvc", "data version control", "data versioning"],
        "data pipelines & etl": ["data pipelines", "data pipeline", "etl", "airflow", "prefect"],
        "automated ci/cd with github actions": ["ci/cd", "continuous integration", "github actions",
                                                "automation pipeline"],
        "grafana & prometheus metrics dashboards": ["grafana", "prometheus", "monitoring dashboard", "metrics"],
        "drift and latency detection": ["drift", "latency", "drift detection", "concept drift", "data drift"],
        "canary releases": ["canary", "shadow rollout", "deployment strategy", "blue-green"],

        # NLP / LLM Specialist
        "regular expressions (regex)": ["regex", "regular expressions", "string matching", "text patterns"],
        "tokenization & lemmatization": ["tokenization", "lemmatization", "stemming", "tokens"],
        "parsing strings": ["parsing", "string parsing", "text manipulation"],
        "spacy and nltk": ["spacy", "nltk", "natural language toolkit", "nlp libraries"],
        "recurrent neural networks (rnns) & lstms": ["rnn", "rnns", "lstm", "lstms", "sequence models"],
        "attention mechanism": ["attention", "attention mechanism", "self-attention", "query key value"],
        "transformer architectures": ["transformer", "transformers", "bert", "gpt", "attention is all you need"],
        "huggingface pre-trained models": ["huggingface", "hf", "pre-trained", "model hub"],
        "prompt patterns": ["prompt patterns", "few-shot", "zero-shot", "cot", "chain of thought"],
        "rag vector search": ["rag", "vector search", "retrieval-augmented generation", "vector db", "chroma",
                              "pinecone"],
        "llamaindex & langchain": ["llamaindex", "langchain", "agent frameworks"],
        "document intelligence & parsing": ["document intelligence", "document parsing", "pdf parsing", "layoutlm"],
        "supervised fine-tuning (sft)": ["supervised fine-tuning", "sft", "instruction tuning"],
        "parameter-efficient fine-tuning (peft/lora)": ["peft", "lora", "qlora", "parameter-efficient", "adapter"],
        "direct preference optimization (dpo)": ["dpo", "direct preference optimization", "alignment"],
        "rlhf alignment": ["rlhf", "pao", "human feedback", "alignment", "reward model"],
        "high-performance llm serving (vllm/ollama)": ["vllm", "ollama", "llm serving", "quantized serving",
                                                       "throughput"],
        "guardrails frameworks (nemo guardrails)": ["guardrails", "nemo guardrails", "safety layer", "moderation"],
        "quantization techniques (awq/gptq)": ["quantization", "awq", "gptq", "bitsandbytes", "4-bit", "8-bit"],
        "model evaluations": ["evaluations", "model evaluation", "benchmark", "mmlu", "evals"],

        # Computer Vision Engineer
        "pixel array manipulation": ["pixel array", "pixel", "pixels", "numpy images", "image arrays"],
        "kernel filters": ["kernel", "filters", "sobel", "gaussian blur", "convolution"],
        "spatial coordinate systems": ["spatial", "coordinate", "coordinates", "3d coordinates", "calibration"],
        "2d/3d geometry": ["geometry", "2d/3d", "transformation", "rotations", "translation"],
        "convolutional neural networks (cnns)": ["cnn", "cnns", "convolutional neural network"],
        "resnet/efficientnet": ["resnet", "efficientnet", "vgg", "backbone", "image classification"],
        "pytorch image classification": ["pytorch image", "torchvision", "image classification pytorch"],
        "data augmentation": ["data augmentation", "augmentations", "albumentations", "rotation", "flipping"],
        "yolo custom object detection": ["yolo", "yolov8", "object detection", "bounding boxes", "yolov5"],
        "object tracking": ["object tracking", "sort", "deepsort", "tracking"],
        "semantic segmentation (u-net)": ["semantic segmentation", "segmentation", "u-net", "unet"],
        "instance segmentation (mask r-cnn)": ["instance segmentation", "mask r-cnn", "mask rcnn"],
        "embedded cv deployment (raspberry pi/jetson)": ["embedded cv", "raspberry pi", "jetson", "jetson nano",
                                                         "edge devices"],
        "tensorrt optimization & quantization": ["tensorrt", "quantization", "int8", "fp16", "onnx", "openvino"],
        "vlm live clients integration": ["vlm live", "vlm client", "vision-language", "multimodal integration"],
        "locomotion sdks interface": ["locomotion", "sdk interface", "robotic locomotion", "unitree sdk",
                                      "sportclient"],
        "clip multimodal embeddings": ["clip", "multimodal embeddings", "contrastive language-image"],
        "stable diffusion pipeline fine-tuning": ["stable diffusion", "diffusion", "image generation",
                                                  "generative vision"],
        "visual grounding": ["visual grounding", "phrase localization", "grounding"],
        "generative vision models": ["generative vision", "diffusion models", "gans"],

        # ML Research Scientist
        "advanced linear algebra": ["linear algebra", "eigenvalues", "eigenvectors", "svd", "matrix decomposition"],
        "multivariate calculus": ["calculus", "gradients", "jacobian", "hessian", "partial derivatives"],
        "mathematical optimization theory": ["optimization", "convex", "non-convex", "gradient descent", "sgd",
                                             "lagrangian"],
        "convex optimization": ["convex optimization", "convex programming", "duality"],
        "derivations of backpropagation": ["backpropagation math", "derivations of backpropagation", "chain rule math"],
        "convergence of optimization algorithms": ["convergence", "sgd convergence", "learning rates",
                                                   "learning rate bounds"],
        "generalization bounds": ["generalization", "generalization bounds", "rademacher complexity", "vc dimension"],
        "statistical learning theory": ["statistical learning", "learning theory", "pac learning"],
        "research paper replication": ["replication", "replicating", "reproduce research", "paper implementation"],
        "jax programming tutorials": ["jax", "grad", "jit", "vmap", "pmap"],
        "autodiff and pytrees": ["autodiff", "pytrees", "automatic differentiation"],
        "custom backpropagation kernels": ["custom kernels", "triton kernels", "jax kernels"],
        "score-based generative models (diffusion mathematics)": ["diffusion math", "score-based", "sde",
                                                                  "diffusion equations"],
        "transformer scaling laws": ["scaling laws", "chinchilla", "parameter scaling", "compute scaling"],
        "empirical scaling law calculations": ["empirical scaling", "scaling laws calculations"],
        "theoretical complexity bounds": ["complexity bounds", "theoretical bounds", "compute complexity"],
        "mechanistic interpretability of model weights": ["mechanistic interpretability", "weight analysis",
                                                          "neel nanda", "circuits"],
        "safety validation frameworks": ["safety validation", "validation frameworks", "alignment bounds"],
        "rlhf mathematical alternatives": ["rlhf math", "dpo math", "reinforcement learning math"],
        "dpo bounds": ["dpo bounds", "direct preference optimization math"]
    }

    completed_topics_found = []
    prioritized_nodes_found = []

    # Simple keywords indicating mastery
    completion_indicators = ["mastered", "know", "confident", "learned", "done", "finish", "completed", "understand",
                             "study", "already", "experienced", "did", "passed"]
    is_mastery_intent = any(ind in text_lower for ind in completion_indicators)

    # Keywords for prioritization
    prioritization_indicators = ["prioritize", "focus", "learn first", "interested in", "start with", "prefer",
                                 "want to learn", "priority"]
    is_prio_intent = any(ind in text_lower for ind in prioritization_indicators)

    for nid, node in nodes.items():
        title_lower = node["title"].lower()
        topics = node.get("topics", [])

        # Check topic-level completion
        for topic in topics:
            topic_lower = topic.lower()
            keywords = TOPIC_KEYWORDS_MAP.get(topic_lower, [topic_lower])
            # If the user expresses having completed/learned this specific topic:
            if any(k in text_lower for k in keywords) and is_mastery_intent:
                if topic not in node["completed_topics"]:
                    node["completed_topics"].append(topic)
                    completed_topics_found.append(topic)

        # Update node status based on topic progress
        total_topics = len(topics)
        completed_count = len(node["completed_topics"])
        node_progress = round((completed_count / total_topics) * 100, 1) if total_topics > 0 else 0.0
        node["progress_percentage"] = node_progress

        if completed_count == total_topics and total_topics > 0:
            node["status"] = "completed"
            node["priority"] = "low"  # Automatically clear high priority when completed
        else:
            node["status"] = "pending"

        # Check node prioritization
        is_prioritized = False
        if is_prio_intent:
            # Match priority based on title words, topic names, or topic keywords in text_lower
            title_words = [w.strip(",.&()") for w in title_lower.split() if len(w.strip(",.&()")) > 2]
            # Match keywords of topics
            topic_keywords = []
            for t in topics:
                topic_keywords.extend(TOPIC_KEYWORDS_MAP.get(t.lower(), [t.lower()]))

            # Match other general abbreviations
            additional_keywords = []
            if "mlops" in title_lower:
                additional_keywords.extend(["mlop", "mlops"])
            if "nlp" in title_lower or "llm" in title_lower:
                additional_keywords.extend(["nlp", "llm", "llms"])
            if "vision" in title_lower:
                additional_keywords.extend(["cv", "vision", "image"])
            if "deployment" in title_lower or "deploy" in title_lower:
                additional_keywords.extend(["deploy", "deployment", "depoly", "depolyment"])

            all_node_keywords = [title_lower] + title_words + [t.lower() for t in
                                                               topics] + topic_keywords + additional_keywords

            # Check if any of our node keywords are present in the user's text
            if any(k in text_lower for k in all_node_keywords if len(k) > 1):
                is_prioritized = True

        if is_prioritized and node["status"] != "completed":
            node["priority"] = "high"
            prioritized_nodes_found.append(node["title"])
        elif is_prio_intent and node["status"] != "completed":
            node["priority"] = "normal"  # Demote previously high-priority nodes if not prioritized now

    # Fallback prioritization if no custom priority matches and there are pending nodes
    if not prioritized_nodes_found:
        for nid in sorted(nodes.keys(), key=lambda x: int(x.split("-")[1])):
            if nodes[nid]["status"] == "pending":
                nodes[nid]["priority"] = "high"
                prioritized_nodes_found.append(nodes[nid]["title"])
                break

    # Persist nested dictionary back to state
    tool_context.state["learning_nodes"] = nodes

    # Re-order nodes: High first, then Normal, then Low, then Completed last
    nodes_list = list(nodes.values())

    def get_sort_key(node):
        if node["status"] == "completed":
            return 4
        prio = node["priority"]
        if prio == "high":
            return 1
        if prio == "normal":
            return 2
        return 3

    nodes_list.sort(key=get_sort_key)

    total_all_topics = sum(len(n.get("topics", [])) for n in nodes.values())
    completed_all_topics = sum(len(n.get("completed_topics", [])) for n in nodes.values())
    overall_progress_pct = round((completed_all_topics / total_all_topics) * 100, 1) if total_all_topics > 0 else 0.0

    print(
        f"\n[ROADMAP ENGINE] Rearranged path by user preferences. Topics completed: {completed_topics_found}, Priorities: {prioritized_nodes_found}. Overall Progress: {overall_progress_pct}%")

    return {
        "status": "success",
        "overall_progress_percentage": overall_progress_pct,
        "completed_topics_count": completed_all_topics,
        "total_topics_count": total_all_topics,
        "rearranged_nodes": nodes_list,
        "message_summary": f"Programmatically prioritized: {prioritized_nodes_found}. Checked off as mastered: {completed_topics_found}."
    }


def mark_node_status(
        node_id: str,
        status: Optional[Literal["pending", "completed"]] = None,
        priority: Optional[Literal["high", "normal", "low"]] = None,
        completed_topics: Optional[List[str]] = None,
        completion_action: Literal["add", "remove", "replace"] = "add",
        tool_context: ToolContext = ToolContext
) -> Dict[str, Any]:
    """
    TASK MUTABILITY: Allows checking off a node or its individual topics as mastered,
    or dynamically changing its learning priority. Returns the recalculated progress.
    """
    if "learning_nodes" not in tool_context.state or not tool_context.state.get("roadmap_initialized", False):
        initialize_student_roadmap(tool_context)

    nodes = tool_context.state["learning_nodes"]

    if node_id not in nodes:
        return {
            "status": "error",
            "message": f"Node ID '{node_id}' does not exist in the active roadmap. Valid IDs are: {list(nodes.keys())}"
        }

    # Apply updates
    if priority:
        nodes[node_id]["priority"] = priority

    if completed_topics is not None:
        valid_topics = nodes[node_id].get("topics", [])
        current_completed = list(nodes[node_id].get("completed_topics", []))
        if completion_action == "add":
            for t in completed_topics:
                if t in valid_topics and t not in current_completed:
                    current_completed.append(t)
        elif completion_action == "remove":
            for t in completed_topics:
                if t in current_completed:
                    current_completed.remove(t)
        elif completion_action == "replace":
            current_completed = [t for t in completed_topics if t in valid_topics]
        nodes[node_id]["completed_topics"] = current_completed
    elif status == "completed":
        # Mark all topics as completed
        nodes[node_id]["completed_topics"] = list(nodes[node_id].get("topics", []))
    elif status == "pending" and completed_topics is None:
        # Clear completed topics
        nodes[node_id]["completed_topics"] = []

    # Update status based on topic completion
    total_topics = len(nodes[node_id].get("topics", []))
    completed_count = len(nodes[node_id].get("completed_topics", []))
    node_progress = round((completed_count / total_topics) * 100, 1) if total_topics > 0 else 0.0
    nodes[node_id]["progress_percentage"] = node_progress

    if completed_count == total_topics and total_topics > 0:
        nodes[node_id]["status"] = "completed"
        nodes[node_id]["priority"] = "low"  # Automatically clear high priority when completed
    else:
        nodes[node_id]["status"] = "pending"

    # Persist updates
    tool_context.state["learning_nodes"] = nodes

    # Recalculate overall progress metrics
    total_all_topics = sum(len(n.get("topics", [])) for n in nodes.values())
    completed_all_topics = sum(len(n.get("completed_topics", [])) for n in nodes.values())
    overall_progress_pct = round((completed_all_topics / total_all_topics) * 100, 1) if total_all_topics > 0 else 0.0

    # Re-order nodes: High first, then Normal, then Low, then Completed last
    nodes_list = list(nodes.values())

    def get_sort_key(node):
        if node["status"] == "completed":
            return 4
        prio = node.get("priority", "normal")
        if prio == "high":
            return 1
        if prio == "normal":
            return 2
        return 3

    nodes_list.sort(key=get_sort_key)

    print(
        f"\n[ROADMAP ENGINE] Node '{node_id}' updated. Topics completed: {nodes[node_id]['completed_topics']}. Overall Progress: {overall_progress_pct}%")

    return {
        "status": "success",
        "overall_progress_percentage": overall_progress_pct,
        "completed_topics_count": completed_all_topics,
        "total_topics_count": total_all_topics,
        "node_updated": nodes[node_id],
        "rearranged_nodes": nodes_list,
        "message": f"Successfully updated Node '{node_id}' ('{nodes[node_id]['title']}') progress to {node_progress}%."
    }


async def aws_browser_search(query: str) -> str:
    """Search the live web for recent information, news, or current facts using the official AWS Bedrock AgentCore cloud-managed browser.
    Args:
        query (str): The search query string or direct URL to look up.
    """
    if not browser_session or not async_playwright:
        return "AWS Managed Browser is not installed in the local environment. Please install bedrock-agentcore and playwright."

    aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    try:
        print(f"\\n[AWS] Initiating remote browser session in {aws_region}...")
        with browser_session(aws_region) as client:
            ws_url, headers = client.generate_ws_headers()
            async with async_playwright() as p:
                print("[AWS] Connecting Playwright over CDP to remote cloud browser...")
                browser = await p.chromium.connect_over_cdp(ws_url, headers=headers)
                print("[AWS] Opening a clean remote page...")
                page = await browser.new_page()
                if query.startswith("http://") or query.startswith("https://"):
                    target_url = query
                    print(f"[AWS] Navigating directly to: {target_url}")
                else:
                    target_url = f"https://www.duckduckgo.com/search?q={query}"
                    print(f"[AWS] Searching for: '{query}'")
                await page.goto(target_url, wait_until="domcontentloaded", timeout=15000)
                raw_text = await page.locator("body").inner_text()
                await browser.close()
                return raw_text[:4000] if raw_text.strip() else "The page loaded but returned no text."
    except Exception as e:
        print(f"\\n[AWS ERROR] Async Browser Tool Failed: {str(e)}\\n")
        return f"Error executing AWS Browser Tool: {str(e)}"


def consult_node_details(node_id: str, tool_context: ToolContext = ToolContext) -> Dict[str, Any]:
    """
    PULLS SYSTEM RESOURCES: Provides the student with detailed explanations and
    structured beginner, intermediate, and advanced learning materials for a specific skill node.
    """
    if "learning_nodes" not in tool_context.state or not tool_context.state.get("roadmap_initialized", False):
        initialize_student_roadmap(tool_context)

    nodes = tool_context.state["learning_nodes"]

    if node_id not in nodes:
        return {
            "status": "error",
            "message": f"Node ID '{node_id}' does not exist in the active roadmap. Valid IDs are: {list(nodes.keys())}"
        }

    selected_node = nodes[node_id]
    print(f"\n[ROADMAP ENGINE] Retrieved resources and details for Node '{node_id}' ('{selected_node['title']}').")

    return {
        "status": "success",
        "node_id": selected_node["id"],
        "title": selected_node["title"],
        "description": selected_node["description"],
        "current_status": selected_node["status"],
        "current_priority": selected_node["priority"],
        "topics": selected_node.get("topics", []),
        "resources": selected_node["resources"],
        "instruction": "Introduce the skill node concept concisely, outlining all of its specific sub-topics. Display the beginner, intermediate, and master learning materials clearly as recommended guides."
    }


# ----------------------------------------------------------------------
# 3. CONFIGURE THE LLM ROADMAP AGENT
# ----------------------------------------------------------------------


def prioritize_node(
        node_id: str,
        priority: Literal["high", "normal", "low"],
        tool_context: ToolContext = ToolContext
) -> Dict[str, Any]:
    """
    PRIORITIZATION ENGINE: Explicitly sets the priority of a specific roadmap milestone.
    Re-orders the roadmap checklist accordingly.
    """
    if "learning_nodes" not in tool_context.state or not tool_context.state.get("roadmap_initialized", False):
        initialize_student_roadmap(tool_context)

    nodes = tool_context.state["learning_nodes"]

    if node_id not in nodes:
        return {
            "status": "error",
            "message": f"Node ID '{node_id}' does not exist in the active roadmap. Valid IDs are: {list(nodes.keys())}"
        }

    nodes[node_id]["priority"] = priority
    tool_context.state["learning_nodes"] = nodes

    # Re-order nodes: High first, then Normal, then Low, then Completed last
    nodes_list = list(nodes.values())

    def get_sort_key(node):
        if node.get("status") == "completed":
            return 4
        prio = node.get("priority", "normal")
        if prio == "high":
            return 1
        if prio == "normal":
            return 2
        return 3

    nodes_list.sort(key=get_sort_key)

    total_all_topics = sum(len(n.get("topics", [])) for n in nodes.values())
    completed_all_topics = sum(len(n.get("completed_topics", [])) for n in nodes.values())
    overall_progress_pct = round((completed_all_topics / total_all_topics) * 100, 1) if total_all_topics > 0 else 0.0

    return {
        "status": "success",
        "overall_progress_percentage": overall_progress_pct,
        "completed_topics_count": completed_all_topics,
        "total_topics_count": total_all_topics,
        "rearranged_nodes": nodes_list,
        "message": f"Successfully updated Node '{node_id}' ('{nodes[node_id]['title']}') priority to {priority}."
    }


SYSTEM_INSTRUCTION = """You are the "AI Career Roadmap Agent", a highly structured, supportive, and adaptive technical mentor.

Your objective is to help students translate their recommended AI career track into a customized, step-by-step learning progression, adapting to their skills, resources, and weekly masteries over time.

## CRITICAL MULTI-TOOL EXECUTION STATE SAFETY RULE:
- You are STRICTLY PROHIBITED from calling multiple state-modifying tools (such as `initialize_student_roadmap`, `rearrange_roadmap_by_preferences`, or `mark_node_status`) on the SAME turn. Calling multiple state-modifying tools in parallel creates a state race condition that wipes out previously mastered/completed topics!
- If the student describes their skills or preferences, ONLY call `rearrange_roadmap_by_preferences(user_preference_text=...)` on that turn. Do NOT call `mark_node_status` in parallel.
- If the student declares they completed a topic/node, ONLY call `mark_node_status(node_id=..., completed_topics=[...], status="pending" or "completed")` on that turn. Do NOT call `rearrange_roadmap_by_preferences` in parallel.
- One state-modifying tool call per turn maximum. This is a compulsory rule to maintain state integrity!

## CO-CREATION & FRACTIONAL MUTABILITY PROCESS (THE STATIC-ROADMAP KILLER)
1. INITIALIZATION: Always start by programmatically calling the `initialize_student_roadmap` tool to fetch their specific track and standardized checklists.
2. CHECK EXISTING SKILLS & FRACTIONAL PRIORITIZATION:
   - Present the checklist milestones as a beautiful, clean numbered list (DO NOT prefix with Node IDs like "Node-1", just use 1, 2, 3, etc.).
   - Each numbered milestone item MUST display:
     * Milestone Title and priority (e.g. High Priority, Normal Priority, Low Priority)
     * Node Progress Percentage and a beautifully formatted 10-character progress bar.
     * Granular sub-topics, clearly labeling which ones are mastered (Done ✅) vs. pending.
   - DO NOT print progress bars for individual topics under any circumstances. Progress bars must ONLY be rendered for each broad milestone/node and for the overall career pathway. Sub-topics should be listed as a simple clean text list with status indicators (Done ✅ vs pending).
   - At the bottom of the list, display a consolidated, overall career path progress bar (e.g. `🏆 Overall Career Path Completion: [██░░░░░░░░] 20.0% completed (4 / 20 topics mastered)`).
   - MANDATORY AUTO-EXECUTION FOR COMPETENCY UPDATES: When a student tells you they have completed, finished, studied, or mastered any topic/concept, you MUST IMMEDIATELY programmatically invoke `mark_node_status` with `completed_topics=[...]` and `status="pending"`. Do NOT ask the student to call the tool, do NOT print instructions on how to call it, and do NOT wait for separate permission. Execute the tool immediately, get the updated state, and show the updated progress bars!
   - Ensure a node only flips to complete ("status": "completed") when ALL of its nested topics are completed. Mastering "Python OOP basics" and "SQL Database queries" should only partially complete Node-1.
   - When a node is completed (100% progress), its priority MUST automatically be demoted to "low" and it must be pushed to the bottom of the active sorted list.

## PROGRESS BAR RENDERING PROTOCOL
For every node/milestone and the overall pathway, render progress bars using a 10-character bar where each '█' represents 10% progress and '░' represents pending progress. You MUST match the shading strictly to the progress_percentage returned by the tool:
- 0%   completed -> `[░░░░░░░░░░]` (Must be completely unshaded!)
- 10%  completed -> `[█░░░░░░░░░]`
- 20%  completed -> `[██░░░░░░░░]`
- 25%  completed -> `[██░░░░░░░░]`
- 30%  completed -> `[███░░░░░░░]`
- 40%  completed -> `[████░░░░░░]`
- 50%  completed -> `[█████░░░░░]`
- 60%  completed -> `[██████░░░░]`
- 70%  completed -> `[███████░░░]`
- 75%  completed -> `[███████░░░]`
- 80%  completed -> `[████████░░]`
- 90%  completed -> `[█████████░]`
- 100% completed -> `[██████████]`
DO NOT hardcode example template progress bars! Always render the 10-character bar dynamically based on the exact percentage returned by the tool.

3. SOURCING NODE DETAILS & BROWSER SEARCHES:
   - When the student asks for details, descriptions, or explanations about any node or topic, you **MUST first execute `consult_node_details`** to fetch catalog metadata.
   - When displaying node details, you **MUST** present the beginner, intermediate, and master resources exactly as they are returned in the `resources` dictionary (e.g., as unified recommendations for the entire milestone/node). Do **NOT** try to split or distribute them across individual sub-topics or write "N/A" for any sub-topic. Present the unified node-level materials clearly!
   - In parallel, you **MUST execute `aws_browser_search`** to retrieve a brief, high-fidelity real-world description of the topic and **append the active real-world resource link at the end of your message**.
   - NEVER fabricate, invent, or output dead or untested URLs (especially GitHub links or documentation sites). If you do not search for a link or the tool doesn't return one, DO NOT include any markdown link. State the book or topic name in bold plain text instead.
4. BOOKS AND PRICING PROTOCOLS:
   - If a student asks for learning books, you **MUST execute `aws_browser_search`** to check active pricing (e.g. in SGD or USD) and purchase places (Amazon, O'Reilly Media, or Kinokuniya Singapore). Provide these verified details clearly.
5. UPDATE PROGRESSION OVER TIME:
   - When the student returns and declares they completed a topic (e.g., "I finished Docker" or "Mark Python as done"), call `mark_node_status` with `completed_topics` updated.
   - Report the updated overall progress percentage and celebrate their upskilling milestones!

## CONVERSATIONAL WORKFLOW
1. Greeting: Welcome the student warmly. Call `initialize_student_roadmap` immediately in the background on Turn 1 to initialize the state machine.
2. Presentation: Write out their matched career pathway recommendation and present their standard roadmap checklists in the clean format (milestones numbered 1-5 with individual node progress bars).
3. Elicit customization: Ask them to share their current skills or what nodes they want to prioritize, or ask about any node they want to learn.
4. Support transitions: Respond step-by-step as the user master nodes or requests guides, always calling the appropriate state machine tool and keeping the overall progress bar in clear view.
"""

# Configure Agent instance for Adaptive Roadmap Assessment
root_agent = Agent(
    model=LiteLlm(model=os.getenv("BEDROCK_MODEL")),
    name="ai_roadmap_agent",
    description="Transforms career outcomes into highly customized, adaptive, and stateful upskilling roadmaps.",
    instruction=SYSTEM_INSTRUCTION,
    tools=[
        initialize_student_roadmap,
        rearrange_roadmap_by_preferences,
        mark_node_status,
        consult_node_details,
        reset_roadmap_session,
        aws_browser_search,
        prioritize_node
    ]
)
