import PyPDF2
import openai
import pandas as pd
from io import BytesIO
import os

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def split_pdf(pdf_path, chunk_size=50):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        total_pages = len(reader.pages)
        chunks = []
        
        for i in range(0, total_pages, chunk_size):
            output = BytesIO()
            writer = PyPDF2.PdfWriter()
            for j in range(i, min(i + chunk_size, total_pages)):
                writer.add_page(reader.pages[j])
            writer.write(output)
            chunks.append(output.getvalue())
    
    return chunks

def extract_text(pdf_chunk):
    reader = PyPDF2.PdfReader(BytesIO(pdf_chunk))
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def process_with_chatgpt(text, prompt, model="gpt-3.5-turbo"):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ],
            max_tokens=150  # Adjust as needed
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        return f"Error processing chunk: {str(e)}"

def main(pdf_path, prompt, model="gpt-3.5-turbo"):
    chunks = split_pdf(pdf_path)
    results = []
    
    for i, chunk in enumerate(chunks):
        text = extract_text(chunk)
        result = process_with_chatgpt(text, prompt, model)
        results.append({"Chunk": i+1, "Result": result})
    
    df = pd.DataFrame(results)
    df.to_csv("results.csv", index=False)
    print("Processing complete. Results saved to results.csv")

if __name__ == "__main__":
    pdf_path = "your_large_pdf.pdf"
    prompt = "I have a list of companies in PDF format, each associated with specific details such as Sr. No., File No., PSN, NIC, Manufacturing Activity, Factory Name, Gala No./Plot No., Industrial Estate, Landmark, Village/M-corp, Tahsil, Dist, Pincode, and the number of Male, Female, and Total Workers. The task is to shortlist companies that Orangewood Labs can potentially collaborate with for deploying 6DoF robotic arms.

Orangewood Labs manufactures 6DoF robotic arms that can be used in applications like powder coating, palletizing, machine tending, and spray painting. Therefore, identify companies from the list whose manufacturing activities align with these applications.

Additionally, exclude any companies with fewer than 20 total workers, as they are less likely to benefit from automation.

Please provide a shortlist of companies that meet these criteria, along with relevant details like Factory Name, Manufacturing Activity, Total Workers, and any other pertinent information. Provide it in tabular form.:"
    model = "gpt-3.5-turbo"  # You can change this to "gpt-4" if you have access
    main(pdf_path, prompt, model)
