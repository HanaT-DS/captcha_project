from webscraping.browser import BrowserConfig, create_browser_and_context, close_browser

cfg = BrowserConfig(headless=False)
pw, browser, context = create_browser_and_context(cfg)

page = context.new_page()
page.goto("https://example.com")
print(page.title())
page.wait_for_timeout(5000)
close_browser(pw, browser, context)
