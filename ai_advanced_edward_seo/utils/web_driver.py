import os
import platform

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
from pathlib import Path

def install_chrome_driver():
    installed_driver = ChromeDriverManager().install()
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_experimental_option(
        "prefs", {
            "profile.managed_default_content_settings.images": 2,
        }
    )
    
    driver_options = options
    driver_basepath = Path(installed_driver).parent
    driver_path = driver_basepath / 'chromedriver'

    if os.path.exists(f'{driver_basepath}/THIRD_PARTY_NOTICES.chromedriver'):
        os.unlink(f'{driver_basepath}/THIRD_PARTY_NOTICES.chromedriver')
    if os.path.exists(f'{driver_basepath}/LICENSE.chromedriver'):
        os.unlink(f'{driver_basepath}/LICENSE.chromedriver')
    os.system(f'chmod +x {driver_path}')
    return driver_basepath, driver_path, driver_options

def reinstall_chrome_driver():
    cur_os = platform.system()
    if cur_os == 'Darwin':
        os.system('brew reinstall chromedriver')
    elif cur_os == 'Linux':
        os.system('sudo apt-get install --reinstall chromedriver')
    elif cur_os == 'Windows':
        os.system('choco reinstall chromedriver')
    else:
        raise ValueError('Unsupported operating system: {cur_os}')
    
def check_chrome_driver():
    try:
        driver_basepath, driver_path, driver_options = install_chrome_driver()
        chrome_driver = webdriver.Chrome(service=Service(str(driver_path)), options=driver_options)
    except WebDriverException as e:
        os.unlink(f'{driver_basepath}/chromedriver')
        driver_basepath, driver_path, driver_options = install_chrome_driver()
        chrome_driver = webdriver.Chrome(service=Service(str(driver_path)), options=driver_options)
    return chrome_driver