from webscraping.scraper import run_scraper

URLS = [
    "https://2captcha.com/fr/demo/normal",
    "https://www.google.com/recaptcha/api2/demo",
    "https://demo.turnstile.workers.dev/",
]

if __name__ == "__main__":
    run_scraper(URLS, headless=False)
