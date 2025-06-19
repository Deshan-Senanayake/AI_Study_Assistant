def generate_flashcards(llm, docs, num=5):
    selected_text = "\n\n".join(doc.page_content for doc in docs[:num])
    prompt = f"""Generate {num} flashcards (Question & Answer format) from the following content:\n\n{selected_text}\n\nFormat:
Q1:
A1:
Q2:
A2:
...
"""
    return llm(prompt)
