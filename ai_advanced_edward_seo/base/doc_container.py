class DocContainer:
    def __init__(self, url: str, langchain_doc=None):
        self.url = url
        self.langchain_doc = langchain_doc

    def get_page_content(self):
        if self.langchain_doc is None:
            return None
        return self.langchain_doc.page_content
