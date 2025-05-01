import time
import random
import requests
from bs4 import BeautifulSoup
import csv
import tqdm

def get_reviews(url):
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')
    
    reviews = []
    for section in soup.select('section.styles_reviewContentwrapper__K2aRu'):
        rating_div = section.select_one('div[data-service-review-rating]')
        rating = int(rating_div['data-service-review-rating']) if rating_div else None
        
        content_el = section.select_one(
            'div.styles_reviewContent__tuXiN '
            'p[data-service-review-text-typography]'
        )
        content = content_el.get_text(strip=True) if content_el else ''
        
        reviews.append({
            'rating': rating,
            'content': content
        })
    return reviews

def scrape_and_save(base_url, max_pages=1000, output_file='reviews.csv'):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['rating', 'content'])
        writer.writeheader()
        
        for page in tqdm.tqdm(range(1, max_pages + 1)):
            url = f"{base_url}?page={page}"
            try:
                reviews = get_reviews(url)
            except requests.HTTPError as e:
                print(f"Erreur HTTP à la page {page} : {e}")
                break
            
            writer.writerows(reviews)  
            
            time.sleep(random.uniform(1, 3))
            
    print(f"Terminé ✓")

if __name__ == '__main__':
    BASE_URL = 'https://www.trustpilot.com/review/www.quicken.com'
    scrape_and_save(BASE_URL, max_pages=2152, output_file='quicken_reviews.csv')