# 🗂️ Directory Structure Generator API

This project provides an intelligent API that generates standardized project directory structures based on a project description, technology stack, and developer preferences. It leverages a Large Language Model (LLM) via the [GROQ API](https://groq.com/), similarity matching using `sentence-transformers`, and is built using FastAPI.

---

## 🚀 Features

- Generate best-practice directory structures for any tech stack.
- Accepts detailed user preferences like including Docker, CI/CD, docs, tests, and custom folders.
- Uses GROQ's LLM for intelligent generation.
- Embedding-based similarity with example projects.
- JSON + Tree view output.
- REST API powered by FastAPI.

---

## 📦 Requirements

- Python 3.9+
- GROQ API Key
- Docker (optional)

---

## 🧪 Example API Request

### Endpoint
`POST /generate`

### Sample Request JSON

```json
{
  "project_desc": "A web application with React frontend and Express backend",
  "tech_stack": ["React", "Node.js", "Express"],
  "preferences": "include docker\ninclude tests\nfolder: uploads"
}
````

### Sample Response

```json
{
  "json_structure": {
    "name": "project-name",
    "structure": [
      {
        "type": "file",
        "name": "README.md"
      },
      ...
    ]
  },
  "tree_view": "project-name/\n├── README.md\n├── .gitignore\n..."
}
```

---

## 🛠️ Installation (Local Development)

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/dir-structure-agent.git
cd dir-structure-agent
```

2. **Create virtual environment & install dependencies**

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Set your GROQ API key**

```bash
export GROQ_API_KEY=your_groq_api_key  # Or use a .env file
```

4. **Run the server**

```bash
uvicorn fastapi_main:app --reload
```

5. **Access the API**
   Visit: [http://localhost:8000/docs](http://localhost:8000/docs) (Swagger UI)

---

## 🐳 Running with Docker

1. **Build Docker image**

```bash
docker build -t dir-structure-agent .
```

2. **Run Docker container**

```bash
docker run -p 8000:8000 --env GROQ_API_KEY=your_groq_api_key dir-structure-agent
```

3. **API is available at**
   [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📁 Project Structure

```
.
├── main.py                # Core logic: agent, preferences, generation
├── fastapi_main.py        # FastAPI endpoint
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🧠 Powered By

* [FastAPI](https://fastapi.tiangolo.com/)
* [GROQ LLM](https://groq.com/)
* [Sentence Transformers](https://www.sbert.net/)
* [TinyDB](https://tinydb.readthedocs.io/)
* [Uvicorn](https://www.uvicorn.org/)

---

## 📝 License

This project is licensed under the MIT License.

---

## 🙋‍♂️ Author

**Aayush Gid**
B.Tech ECE | Embedded Systems & AI Enthusiast
[LinkedIn](https://linkedin.com/in/your-profile) | [GitHub](https://github.com/yourusername)

---

## 🤝 Contributions

Contributions, suggestions, and feature requests are welcome!
Open a pull request or an issue anytime.