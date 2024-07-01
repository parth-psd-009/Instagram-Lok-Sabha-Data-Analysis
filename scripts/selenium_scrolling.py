import time
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, WebDriverException

crowdtangle_url = "https://apps.crowdtangle.com/instagrampoliticalmonitoring/lists#"
driver = webdriver.Chrome(ChromeDriverManager().install())

try:
    driver.get(crowdtangle_url)
    
    auth = driver.find_element(By.CLASS_NAME, 'facebookLoginButton__authButton--lof0c')
    auth.click()
    time.sleep(5)
    
    handles = driver.window_handles
    driver.switch_to.window(handles[1])
    time.sleep(5)
    
    email_input = driver.find_element(By.ID, 'email')
    password_input = driver.find_element(By.ID, 'pass')
    
    email_input.send_keys('khushalgoyal77@gmail.com')  
    password_input.send_keys('chanie24')  
    
    auth_submit_parent = driver.find_element(By.ID, 'loginbutton')
    auth_submit = auth_submit_parent.find_element(By.TAG_NAME, 'input')
    auth_submit.click()
    
    time.sleep(15)
    driver.switch_to.window(handles[0])
    time.sleep(10)
    
    element = driver.find_element(By.ID, 'body-container')
    
    max_scrolls = 500
    scroll_count = 0
    
    while scroll_count < max_scrolls:
        try:
            driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", element)
            print('Scrolling Down')
            scroll_count += 1
            time.sleep(min(10 * 1.25**scroll_count, 90))
        except WebDriverException as e:
            print(f"Error occurred: {e}")
            break
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    driver.quit()
