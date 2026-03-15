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
        # Initialize a Pinecone client with your API key
        self.pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        self.index_name = index_name
        cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
        region = os.environ.get('PINECONE_REGION') or 'us-east-1'
        spec = ServerlessSpec(cloud=cloud, region=region)

        if not self.pc.has_index(index_name):
            self.pc.create_index(
                name=index_name,
                dimension=embeddings.dimension,
                metric="cosine",
                spec=spec
            )
        self.index = self.pc.Index(index_name)

    def vectorize_term_json(self, json_data: Dict):
        print('here!')
        course_numbers = list(json_data.keys())
        class_datas = list(json_data.values())

        print(course_numbers)

        class_dict_info = [self._class_to_text(cd, format='dict') for cd in class_datas]
        class_text_info = [self._class_to_text(cd, format='text') for cd in class_datas]

        print(class_dict_info[0])

        batch_size = 50
        total = len(class_text_info)
        for i in range(0, total, batch_size):
            batch_numbers = course_numbers[i:i + batch_size]
            batch_dicts = class_dict_info[i:i + batch_size]
            batch_texts = class_text_info[i:i + batch_size]

            vectors = embeddings.embed_documents(batch_texts)

            records = []
            for course_number, metadata, vector in zip(batch_numbers, batch_dicts, vectors):
                records.append({
                    "id": f"{course_number}",
                    "values": vector,
                    "namespace": 'course-catalog',
                    "metadata": metadata,
                })

            self.upsert_batch(records)
            print(f"Upserted {min(i + batch_size, total)}/{total} courses")
            if i + batch_size < total:
                time.sleep(5)

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

        return_results = []
        for match in results.matches:
            m = match.metadata.copy()
            m['score'] = match.score
            return_results.append(m)
        return return_results

    def _class_to_text(self, class_data: Dict, format: str) -> tuple:
        parts = {}
        
        if class_data.get('number'):
            parts['Class Number'] = str(class_data['number'])
        if class_data.get('subject'):
            parts['Class Subject'] = str(class_data['subject'])
        if class_data.get('name'):
            parts['Class Name'] = class_data['name']
        if class_data.get('description'):
            parts['Class Description'] = class_data['description']
        if class_data.get('inCharge'):
            parts['Class Instructors'] = class_data['inCharge']

        if class_data.get('course'):
            parts['Course Number'] = str(class_data['course'])

        if class_data.get('rating'):
            parts['Class Rating'] = str(class_data['rating'])
        if class_data.get('hours'):
            parts['Class Hours'] = str(class_data['hours'])
        if class_data.get('size'):
            parts['Class Size'] = str(class_data['size'])

        if class_data.get('final'):
            parts['Class Has Final'] = str(class_data['final'])
        if class_data.get('half'):
            parts['Class is Half-Semester'] = str(class_data['half'])
        if class_data.get('limited'):
            parts['Class is Limited Enrollment'] = str(class_data['limited'])
        if class_data.get('new'):
            parts['Class is New'] = str(class_data['new'])

        if class_data.get('terms'):
            parts['Terms Offered'] = ', '.join(class_data['terms'])
        if class_data.get('prereqs'):
            parts['Prerequisites'] = ', '.join(class_data['prereqs'])
        if class_data.get('sectionKinds'):
            parts['Section Kinds'] = ', '.join(class_data['sectionKinds']) 

        if class_data.get('lectureRawSections'):
            parts['Raw Data for Lecture Sections'] = str(class_data['lectureRawSections'])
        if class_data.get('lectureSections'):
            parts['Formatted Data for Lecture Sections'] = str(class_data['lectureSections'])
        if class_data.get('recitationRawSections'):
            parts['Raw Data for Recitation Sections'] = str(class_data['recitationRawSections'])
        if class_data.get('recitationSections'):
            parts['Formatted Data for Recitation Sections'] = str(class_data['recitationSections'])

        units = (class_data.get('lectureUnits', 0) + class_data.get('labUnits', 0) + class_data.get('preparationUnits', 0))
        if units:
            parts['Units'] = str(units)
        
        level = class_data.get('level', '')
        if level:
            parts['Class Level'] = f"{'Undergraduate' if level == 'U' else 'Graduate'}"

        same_class = (class_data.get('same', '') + class_data.get('meets', ''))
        if same_class:
            parts['Class Meets With'] = same_class

        hass = class_data.get('hass', [])
        if hass:
            parts['Class HASS Categories Satisfied'] = f"{'', ', '.join(hass)}"

        cih = class_data.get('comms', [])
        if cih:
            parts['Class CI-H Status'] = f"{'', ', '.join(cih)}" 
        cim = class_data.get('cim', [])
        if cim:
            parts['Class CI-M Status'] = f"{'', ', '.join(cim)}" 

        gir = class_data.get('gir', [])
        if gir:
            parts['Class GIR Categories Satisfied'] = f"{'', ', '.join(gir)}"

        offered = class_data.get('offered', [])
        if offered:
            parts['Terms Class was Offered'] = f"{'', ', '.join(offered)}"
        taken_by = class_data.get('taken_by', [])
        if taken_by:
            parts['Students Who Have Taken This Class'] = f"{'', ', '.join(taken_by)}"
        
        if format == 'dict':
            return parts
        elif format == 'text':  
            return '\n'.join(parts.values())
        return None


def load_term_data(service: PineconeService, filename: str, data_dir: str = 'data'):
    filepath = os.path.join(data_dir, filename)
    with open(filepath) as f:
        json_data = json.load(f)
    service.vectorize_term_json(json_data)