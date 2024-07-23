from langchain.schema.retriever import BaseRetriever
from langchain_core.documents import Document
# from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.vectorstores import VectorStore
from langchain_core.stores import BaseStore 
from langchain_text_splitters.base import TextSplitter
from typing import Any, Dict, List, Optional 

class ParentDocumentRetriever(BaseRetriever):
    vectorstore: VectorStore
    docstore: BaseStore[str, Document]
    id_key: str = "doc_id"
    search_kwargs: dict = dict(default_factory=dict)
    child_splitter: TextSplitter
    parent_splitter: Optional[TextSplitter] = None

    def _get_relevant_documents(
        self,
        query: Dict[str, Any],
        *,
        # run_manager: CallbackManagerForRetrieverRun,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        all_results = []
        if metadata_filter:
            # Iterate over each key-value pair in the metadata_filter
            unique_ids = set()

            # Iterate over each key-value pair in the metadata_filter
            for key, value in metadata_filter.items():
                # Perform the similarity search for the current key-value pair
                sub_docs = self.vectorstore.similarity_search(query, k=10, filter={key: value}, **self.search_kwargs)
                ids = [d.metadata[self.id_key] for d in sub_docs]

                # Add unique document IDs to the set
                unique_ids.update(ids)

            # Retrieve documents from the docstore based on the unique IDs
            all_results = self.docstore.mget(list(unique_ids))
            print("Filtering documents with metadata:", metadata_filter)
            filtered_documents = []

            for document in all_results:
                if document is not None:
                    match = all(
                        any(value in document.metadata.get(key, []) for value in values)
                        if isinstance(document.metadata.get(key), list)
                        else document.metadata.get(key) in values
                        for key, values in metadata_filter.items() if values
                        )
                if match:
                    filtered_documents.append(document)

            docs = filtered_documents
        else:
            sub_docs = self.vectorstore.similarity_search(query, k=10, **self.search_kwargs)
            ids = []
            for d in sub_docs:
                if d.metadata[self.id_key] not in ids:
                    ids.append(d.metadata[self.id_key])
            docs = self.docstore.mget(ids)

        return [d for d in docs if d is not None]

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        add_to_docstore: bool = True,
    ) -> None:
        if self.parent_splitter is not None:
            documents = self.parent_splitter.split_documents(documents)
        if ids is None:
            doc_ids = [str(uuid.uuid4()) for _ in documents]
            if not add_to_docstore:
                raise ValueError(
                    "If ids are not passed in, `add_to_docstore` MUST be True"
                )
        else:
            if len(documents) != len(ids):
                raise ValueError(
                    "Got uneven list of documents and ids. "
                    "If `ids` is provided, should be same length as `documents`."
                )
            doc_ids = ids

        docs = []
        full_docs = []
        for i, doc in enumerate(documents):
            _id = doc_ids[i]
            sub_docs = self.child_splitter.split_documents([doc])
            for _doc in sub_docs:
                _doc.metadata[self.id_key] = _id
            docs.extend(sub_docs)
            full_docs.append((_id, doc))
        self.vectorstore.add_documents(docs)
        if add_to_docstore:
            self.docstore.mset(full_docs)