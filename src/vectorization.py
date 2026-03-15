from typing import Dict, List, cast
import os
import json
import glob
import time
from dotenv import load_dotenv
from langchain_pinecone import PineconeEmbeddings
from pinecone import Pinecone, ServerlessSpec
from pinecone.core.openapi.db_data.model.query_response import QueryResponse

load_dotenv()

model_name = 'multilingual-e5-large'
embeddings = PineconeEmbeddings(
    model=model_name,
    pinecone_api_key=os.environ.get('PINECONE_API_KEY') # pyright: ignore[reportArgumentType]
)


class PineconeService:

    def __init__(self, index_name: str):
        self.pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        self.index_name = index_name
        cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
        region = os.environ.get('PINECONE_REGION') or 'us-east-1'
        spec = ServerlessSpec(cloud=cloud, region=region)

        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=embeddings.dimension,
                metric="cosine",
                spec=spec
            )
        self.index = self.pc.Index(index_name)

    def upsert_class(self, vector: List[float], class_data: Dict, term: str):
        course_number = class_data.get('number', '')
        units = (
            class_data.get('lectureUnits', 0)
            + class_data.get('labUnits', 0)
            + class_data.get('preparationUnits', 0)
        )
        self.index.upsert(
            vectors=[{
                "id": f"{course_number}_{term}",
                "values": vector,
                "metadata": {
                    "course_number": course_number,
                    "name": class_data.get('name', ''),
                    "term": term,
                    "description": class_data.get('description', ''),
                    "prereqs": class_data.get('prereqs', ''),
                    "level": class_data.get('level', ''),
                    "units": units,
                },
            }],
            namespace=term,
        )

    def upsert_batch(self, records):
        if not records:
            return 
        for i in range(0, len(records), 100):
            chunk = records[i:i + 100]
            namespace = chunk[0].get('namespace', '')
            self.index.upsert(
                vectors=[{k: v for k, v in r.items() if k != 'namespace'} for r in chunk], # type: ignore
                namespace=namespace,
            )

    def get_course_by_id(self, course_number: str, namespace: str) -> Dict | None:
        result = cast(QueryResponse, self.index.query(
            id=f"{course_number}_{namespace}",
            namespace=namespace,
            top_k=1,
            include_values=False,
            include_metadata=True,
        ))
        if result.matches:
            return dict(result.matches[0].metadata)
        return None
    
    def query_and_filter(self, query_text: str, filter: dict | None, top_k: int = 5, namespace: str | None = None) -> List[Dict]:
        vector = embeddings.embed_query(query_text)

        if filter:
            results = cast(QueryResponse, self.index.query(
                vector=vector,
                top_k=top_k,
                include_metadata=True,
                filter=filter,
                namespace=namespace,
            ))
        else:
            results = cast(QueryResponse, self.index.query(
                vector=vector,
                top_k=top_k,
                include_metadata=True,
                namespace=namespace,
            ))

        return [
            {
                "course_number": match.metadata.get("course_number"),
                "name": match.metadata.get("name"),
                "term": match.metadata.get("term"),
                "description": match.metadata.get("description"),
                "hours": match.metadata.get("hours"),
                "prereqs": match.metadata.get("prereqs"),
                "units": match.metadata.get("units"),
                "hass":match.metadata.get("hass"),
                "level": match.metadata.get("level"),
                "score": match.score,
            }
            for match in results.matches
        ]

    def query(self, query_text: str, top_k: int = 5, namespace: str | None = None) -> List[Dict]:
        vector = embeddings.embed_query(query_text)
        results = cast(QueryResponse, self.index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace,
        ))

        return [
            {
                "course_number": match.metadata.get("course_number"),
                "name": match.metadata.get("name"),
                "term": match.metadata.get("term"),
                "description": match.metadata.get("description"),
                "hours": match.metadata.get("hours"),
                "prereqs": match.metadata.get("prereqs"),
                "units": match.metadata.get("units"),
                "hass":match.metadata.get("hass"),
                "level": match.metadata.get("level"),
                "score": match.score,
            }
            for match in results.matches
        ]
    
    def vectorize_term_json(self, json_data: Dict):
        term = json_data.get('termInfo', {}).get('urlName', 'unknown')
        classes = json_data.get('classes', {})

        course_numbers = list(classes.keys())
        class_datas = list(classes.values())
        texts = [self._class_to_text(cd) for cd in class_datas]

        batch_size = 50
        total = len(texts)
        for i in range(0, total, batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_numbers = course_numbers[i:i + batch_size]
            batch_datas = class_datas[i:i + batch_size]

            vectors = embeddings.embed_documents(batch_texts)

            records = []
            for course_number, class_data, text, vector in zip(batch_numbers, batch_datas, batch_texts, vectors):
                units = (
                    class_data.get('lectureUnits', 0)
                    + class_data.get('labUnits', 0)
                    + class_data.get('preparationUnits', 0)
                )
                records.append({
                    "id": f"{course_number}_{term}",
                    "values": vector,
                    "namespace": term,
                    "metadata": {
                        "course_number": course_number,
                        "name": class_data.get('name', ''),
                        "term": term,
                        "description": class_data.get('description', ''),
                        "prereqs": class_data.get('prereqs', ''),
                        "level": class_data.get('level', ''),
                        "units": units,
                        "text": text,
                    },
                })

            self.upsert_batch(records)
            print(f"Upserted {min(i + batch_size, total)}/{total} courses")
            if i + batch_size < total:
                time.sleep(5)


    def _class_to_text(self, class_data: Dict) -> str:
        print(class_data)
        parts = []
        if class_data.get('number') and class_data.get('name'):
            parts.append(f"Course: {class_data['number']} - {class_data['name']}")
        if class_data.get('description'):
            parts.append(f"Description: {class_data['description']}")
        prereqs = class_data.get('prereqs', '')
        if prereqs and prereqs != 'None':
            parts.append(f"Prerequisites: {prereqs}")
        hass = class_data.get('hass', [])
        if hass:
            parts.append(f"HASS: {', '.join(hass)}")
        units = (
            class_data.get('lectureUnits', 0)
            + class_data.get('labUnits', 0)
            + class_data.get('preparationUnits', 0)
        )
        if units:
            parts.append(f"Units: {units}")
        level = class_data.get('level', '')
        if level:
            parts.append(f"Level: {'Undergraduate' if level == 'U' else 'Graduate'}")
        return '\n'.join(parts)


def load_term_data(service: PineconeService, filename: str, data_dir: str = 'data'):
    filepath = os.path.join(data_dir, filename)
    with open(filepath) as f:
        json_data = json.load(f)
    service.vectorize_term_json(json_data)


