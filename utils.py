import os
import json
import hashlib
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import re
from datetime import datetime

# Third-party imports
import gradio as gr
from groq import Groq
from sentence_transformers import SentenceTransformer
from tinydb import TinyDB, Query
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProjectPreferences:
    """Data class for project preferences"""
    include_docs: bool = True
    include_tests: bool = True
    include_docker: bool = False
    include_ci_cd: bool = False
    custom_folders: List[str] = None
    framework_specific: bool = True
    
    def __post_init__(self):
        if self.custom_folders is None:
            self.custom_folders = []

class DirectoryStructureAgent:
    """
    Agent that suggests standardized project directory structures based on
    project description, tech stack, team roles, and best practices.
    """
    
    def __init__(self, groq_api_key: str, example_repo_index=None, cache_db_path="cache.json"):
        """
        Initialize the DirectoryStructureAgent
        
        Args:
            groq_api_key: GROQ API key for LLM inference
            example_repo_index: Optional SentenceTransformer model for similarity search
            cache_db_path: Path to cache database file
        """
        self.groq_client = Groq(api_key=groq_api_key)
        self.example_repo_index = example_repo_index
        self.cache_db = TinyDB(cache_db_path)
        
        # Initialize sentence transformer for similarity search (optional)
        try:
            if example_repo_index is None:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            else:
                self.sentence_model = example_repo_index
        except Exception as e:
            logger.warning(f"Could not load sentence transformer: {e}")
            self.sentence_model = None
        
        # Load example repositories for similarity matching
        self.example_repos = self._load_example_repos()
    
    def _load_example_repos(self) -> List[Dict]:
        """Load example repository structures for similarity matching"""
        return [
            {
                "description": "React TypeScript frontend with Node.js backend",
                "tech_stack": ["React", "TypeScript", "Node.js", "Express"],
                "structure": {
                    "name": "fullstack-app",
                    "structure": [
                        {"type": "file", "name": "README.md"},
                        {"type": "file", "name": ".gitignore"},
                        {"type": "file", "name": "package.json"},
                        {"type": "folder", "name": "frontend", "children": [
                            {"type": "file", "name": "package.json"},
                            {"type": "folder", "name": "src", "children": [
                                {"type": "folder", "name": "components", "children": []},
                                {"type": "folder", "name": "pages", "children": []},
                                {"type": "folder", "name": "utils", "children": []},
                                {"type": "file", "name": "App.tsx"},
                                {"type": "file", "name": "index.tsx"}
                            ]},
                            {"type": "folder", "name": "public", "children": []}
                        ]},
                        {"type": "folder", "name": "backend", "children": [
                            {"type": "file", "name": "package.json"},
                            {"type": "folder", "name": "src", "children": [
                                {"type": "folder", "name": "routes", "children": []},
                                {"type": "folder", "name": "models", "children": []},
                                {"type": "folder", "name": "middleware", "children": []},
                                {"type": "file", "name": "server.js"}
                            ]}
                        ]},
                        {"type": "folder", "name": "tests", "children": []},
                        {"type": "folder", "name": "docs", "children": []}
                    ]
                }
            },
            {
                "description": "Python Django REST API with PostgreSQL",
                "tech_stack": ["Python", "Django", "PostgreSQL", "Redis"],
                "structure": {
                    "name": "django-api",
                    "structure": [
                        {"type": "file", "name": "README.md"},
                        {"type": "file", "name": ".gitignore"},
                        {"type": "file", "name": "requirements.txt"},
                        {"type": "file", "name": "manage.py"},
                        {"type": "folder", "name": "app", "children": [
                            {"type": "file", "name": "__init__.py"},
                            {"type": "file", "name": "settings.py"},
                            {"type": "file", "name": "urls.py"},
                            {"type": "file", "name": "wsgi.py"}
                        ]},
                        {"type": "folder", "name": "api", "children": [
                            {"type": "file", "name": "__init__.py"},
                            {"type": "file", "name": "models.py"},
                            {"type": "file", "name": "views.py"},
                            {"type": "file", "name": "serializers.py"},
                            {"type": "file", "name": "urls.py"}
                        ]},
                        {"type": "folder", "name": "tests", "children": []},
                        {"type": "folder", "name": "docs", "children": []}
                    ]
                }
            }
        ]
    
    def _make_cache_key(self, project_desc: str, tech_stack: List[str], prefs: Optional[ProjectPreferences] = None) -> str:
        """Generate a cache key for the given parameters"""
        cache_data = {
            "project_desc": project_desc.lower().strip(),
            "tech_stack": sorted([tech.lower().strip() for tech in tech_stack]),
            "prefs": prefs.__dict__ if prefs else {}
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _find_similar_repos(self, project_desc: str, tech_stack: List[str], top_k: int = 2) -> List[Dict]:
        """Find similar repository examples using sentence similarity"""
        if not self.sentence_model:
            return []
        
        try:
            query_text = f"{project_desc} {' '.join(tech_stack)}"
            query_embedding = self.sentence_model.encode([query_text])
            
            similarities = []
            for repo in self.example_repos:
                repo_text = f"{repo['description']} {' '.join(repo['tech_stack'])}"
                repo_embedding = self.sentence_model.encode([repo_text])
                similarity = np.dot(query_embedding[0], repo_embedding[0]) / (
                    np.linalg.norm(query_embedding[0]) * np.linalg.norm(repo_embedding[0])
                )
                similarities.append((similarity, repo))
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x[0], reverse=True)
            return [repo for _, repo in similarities[:top_k]]
        
        except Exception as e:
            logger.warning(f"Similarity search failed: {e}")
            return []
    
    def _build_prompt(self, project_desc: str, tech_stack: List[str], 
                     prefs: Optional[ProjectPreferences] = None,
                     similar_repos: List[Dict] = None) -> str:
        """Build the prompt for the LLM"""
        
        similar_examples = ""
        if similar_repos:
            similar_examples = "\n\nHere are some similar project examples for reference:\n"
            for i, repo in enumerate(similar_repos, 1):
                similar_examples += f"\nExample {i}:\n"
                similar_examples += f"Description: {repo['description']}\n"
                similar_examples += f"Tech Stack: {', '.join(repo['tech_stack'])}\n"
        
        preferences_text = ""
        if prefs:
            preferences_text = f"""
Additional Requirements:
- Include documentation folder: {prefs.include_docs}
- Include tests folder: {prefs.include_tests}
- Include Docker support: {prefs.include_docker}
- Include CI/CD: {prefs.include_ci_cd}
- Custom folders: {', '.join(prefs.custom_folders) if prefs.custom_folders else 'None'}
- Framework-specific structure: {prefs.framework_specific}
"""
        
        prompt = f"""You are an expert software architect. Create a standardized project directory structure based on the following requirements:

Project Description: {project_desc}
Tech Stack: {', '.join(tech_stack)}
{preferences_text}
{similar_examples}

Generate a comprehensive directory structure that follows best practices for the given tech stack. The output must be a valid JSON object with the following exact schema:

{{
  "name": "project-name",
  "structure": [
    {{"type": "file", "name": "README.md"}},
    {{"type": "file", "name": ".gitignore"}},
    {{"type": "folder", "name": "src", "children": [
      {{"type": "file", "name": "main.py"}},
      {{"type": "folder", "name": "utils", "children": []}}
    ]}}
  ]
}}

Rules:
1. Always include README.md and .gitignore
2. Use appropriate file extensions for the tech stack
3. Follow language/framework conventions
4. Include common configuration files
5. Structure should be logical and scalable
6. Use lowercase names with hyphens or underscores
7. Include only the JSON, no additional text or explanations

Generate the directory structure now:"""
        
        return prompt
    
    def _parse_llm_output(self, llm_response: str) -> Optional[Dict]:
        """Parse the LLM output to extract JSON structure"""
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                # Try to parse the entire response as JSON
                return json.loads(llm_response.strip())
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM output as JSON: {e}")
            logger.error(f"LLM Response: {llm_response}")
            return None
    
    def _validate_structure(self, structure: Dict) -> bool:
        """Validate the generated directory structure"""
        if not isinstance(structure, dict):
            return False
        
        if "name" not in structure or "structure" not in structure:
            return False
        
        if not isinstance(structure["structure"], list):
            return False
        
        # Check for required files
        file_names = []
        for item in structure["structure"]:
            if item.get("type") == "file":
                file_names.append(item.get("name", ""))
        
        required_files = ["README.md", ".gitignore"]
        for req_file in required_files:
            if req_file not in file_names:
                logger.warning(f"Missing required file: {req_file}")
                return False
        
        return True
    
    def _apply_preferences(self, structure: Dict, prefs: Optional[ProjectPreferences]) -> Dict:
        """Apply user preferences to the structure"""
        if not prefs:
            return structure
        
        structure_list = structure["structure"]
        
        # Add custom folders
        for custom_folder in prefs.custom_folders:
            if not any(item.get("name") == custom_folder for item in structure_list):
                structure_list.append({
                    "type": "folder",
                    "name": custom_folder,
                    "children": []
                })
        
        # Add Docker support
        if prefs.include_docker:
            docker_files = ["Dockerfile", "docker-compose.yml", ".dockerignore"]
            for docker_file in docker_files:
                if not any(item.get("name") == docker_file for item in structure_list):
                    structure_list.append({
                        "type": "file",
                        "name": docker_file
                    })
        
        # Add CI/CD support
        if prefs.include_ci_cd:
            if not any(item.get("name") == ".github" for item in structure_list):
                structure_list.append({
                    "type": "folder",
                    "name": ".github",
                    "children": [
                        {
                            "type": "folder",
                            "name": "workflows",
                            "children": [
                                {"type": "file", "name": "ci.yml"}
                            ]
                        }
                    ]
                })
        
        return structure
    
    def _generate_with_groq(self, prompt: str) -> Optional[str]:
        """Generate response using GROQ API"""
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert software architect that generates project directory structures in JSON format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="llama3-8b-8192",  # Using Llama 3 model
                temperature=0.3,
                max_tokens=2048,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            logger.error(f"GROQ API call failed: {e}")
            return None
    
    def suggest_structure(self, project_id: str, project_desc: str, 
                         tech_stack: List[str], prefs: Optional[ProjectPreferences] = None) -> Optional[Dict]:
        """
        Main method to suggest project directory structure
        
        Args:
            project_id: Unique identifier for the project
            project_desc: Description of the project
            tech_stack: List of technologies used in the project
            prefs: Optional preferences for the structure
            
        Returns:
            Dictionary containing the suggested directory structure
        """
        try:
            # Check cache first
            cache_key = self._make_cache_key(project_desc, tech_stack, prefs)
            QueryObj = Query()
            cached_result = self.cache_db.table('structures').search(QueryObj.cache_key == cache_key)
            
            if cached_result:
                logger.info("Returning cached result")
                return cached_result[0]['structure']
            
            # Find similar repositories
            similar_repos = self._find_similar_repos(project_desc, tech_stack)
            
            # Build prompt
            prompt = self._build_prompt(project_desc, tech_stack, prefs, similar_repos)
            
            # Generate structure using GROQ
            llm_response = self._generate_with_groq(prompt)
            if not llm_response:
                logger.error("Failed to get response from GROQ")
                return None
            
            # Parse LLM output
            structure = self._parse_llm_output(llm_response)
            if not structure:
                logger.error("Failed to parse LLM output")
                return None
            
            # Validate structure
            if not self._validate_structure(structure):
                logger.error("Generated structure is invalid")
                return None
            
            # Apply preferences
            structure = self._apply_preferences(structure, prefs)
            
            # Cache the result
            self.cache_db.table('structures').insert({
                'cache_key': cache_key,
                'project_id': project_id,
                'structure': structure,
                'timestamp': datetime.now().isoformat()
            })
            
            return structure
            
        except Exception as e:
            logger.error(f"Error in suggest_structure: {e}")
            return None
    
    def structure_to_tree(self, structure: Dict, indent: str = "") -> str:
        """Convert JSON structure to text tree representation"""
        if not structure:
            return "Invalid structure"
        
        tree_lines = [f"{structure['name']}/"]
        
        def _build_tree(items: List[Dict], current_indent: str) -> List[str]:
            lines = []
            for i, item in enumerate(items):
                is_last = i == len(items) - 1
                prefix = "└── " if is_last else "├── "
                name = item.get('name', 'unnamed')
                
                if item.get('type') == 'folder':
                    lines.append(f"{current_indent}{prefix}{name}/")
                    if 'children' in item and item['children']:
                        next_indent = current_indent + ("    " if is_last else "│   ")
                        lines.extend(_build_tree(item['children'], next_indent))
                else:
                    lines.append(f"{current_indent}{prefix}{name}")
            return lines
        
        if 'structure' in structure:
            tree_lines.extend(_build_tree(structure['structure'], ""))
        
        return "\n".join(tree_lines)

# Initialize the agent (will be done in Gradio interface)
def create_agent():
    """Create DirectoryStructureAgent instance"""
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    
    return DirectoryStructureAgent(groq_api_key)

def parse_preferences(prefs_text: str) -> ProjectPreferences:
    """Parse preferences from text input"""
    prefs = ProjectPreferences()
    
    if not prefs_text.strip():
        return prefs
    
    prefs_lower = prefs_text.lower()
    
    # Parse boolean preferences
    prefs.include_docs = "docs" in prefs_lower or "documentation" in prefs_lower
    prefs.include_tests = "test" in prefs_lower
    prefs.include_docker = "docker" in prefs_lower
    prefs.include_ci_cd = "ci" in prefs_lower or "github actions" in prefs_lower
    
    # Parse custom folders
    custom_folders = []
    lines = prefs_text.split('\n')
    for line in lines:
        if line.strip().startswith('folder:') or line.strip().startswith('custom:'):
            folder_name = line.split(':', 1)[1].strip()
            if folder_name:
                custom_folders.append(folder_name)
    
    prefs.custom_folders = custom_folders
    return prefs

def generate_directory_structure(project_desc: str, tech_stack: str, preferences: str):
    """Generate directory structure for Gradio interface"""
    try:
        # Create agent
        agent = create_agent()
        
        # Parse inputs
        tech_list = [tech.strip() for tech in tech_stack.split(',') if tech.strip()]
        prefs = parse_preferences(preferences)
        
        # Generate structure
        project_id = f"gradio_{hashlib.md5(project_desc.encode()).hexdigest()[:8]}"
        structure = agent.suggest_structure(project_id, project_desc, tech_list, prefs)
        
        if not structure:
            return "Failed to generate directory structure. Please try again.", "Error occurred."
        
        # Format outputs
        json_output = json.dumps(structure, indent=2)
        tree_output = agent.structure_to_tree(structure)
        
        return json_output, tree_output
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return error_msg, error_msg
