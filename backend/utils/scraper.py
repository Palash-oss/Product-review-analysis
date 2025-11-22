"""
Advanced E-commerce Review Scraper
Supports: Amazon, Flipkart, Myntra, and generic e-commerce sites
"""

import re
import asyncio
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs
import aiohttp
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import logging

logger = logging.getLogger(__name__)

class ProductScraper:
    """Universal product and review scraper for major e-commerce platforms"""
    
    def __init__(self):
        self.ua = UserAgent()
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _get_headers(self) -> Dict[str, str]:
        """Generate random headers to avoid blocking"""
        return {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0',
        }
    
    def identify_platform(self, url: str) -> str:
        """Identify which e-commerce platform the URL belongs to"""
        domain = urlparse(url).netloc.lower()
        
        if 'amazon' in domain:
            return 'amazon'
        elif 'flipkart' in domain:
            return 'flipkart'
        elif 'myntra' in domain:
            return 'myntra'
        elif 'ajio' in domain:
            return 'ajio'
        elif 'meesho' in domain:
            return 'meesho'
        else:
            return 'generic'
    
    async def scrape_product(self, url: str) -> Dict:
        """Main entry point - scrapes product details and reviews"""
        platform = self.identify_platform(url)
        logger.info(f"Scraping {platform} product: {url}")
        
        scraper_map = {
            'amazon': self._scrape_amazon,
            'flipkart': self._scrape_flipkart,
            'myntra': self._scrape_myntra,
            'ajio': self._scrape_generic,
            'meesho': self._scrape_generic,
            'generic': self._scrape_generic,
        }
        
        scraper = scraper_map.get(platform, self._scrape_generic)
        return await scraper(url)
    
    async def _fetch_page(self, url: str, retries: int = 3) -> Optional[str]:
        """Fetch page content with retry logic"""
        for attempt in range(retries):
            try:
                if not self.session:
                    self.session = aiohttp.ClientSession()
                    
                async with self.session.get(
                    url,
                    headers=self._get_headers(),
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    if response.status == 200:
                        return await response.text()
                    elif response.status == 404:
                        logger.error(f"Product not found: {url}")
                        return None
                    else:
                        logger.warning(f"Status {response.status} for {url}")
                        
            except Exception as e:
                logger.error(f"Fetch attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    
        return None
    
    async def _scrape_amazon(self, url: str) -> Dict:
        """Scrape Amazon product details and reviews"""
        html = await self._fetch_page(url)
        if not html:
            return self._empty_result("Failed to fetch Amazon page")
        
        soup = BeautifulSoup(html, 'lxml')
        
        # Extract product details
        product_name = self._extract_amazon_title(soup)
        price = self._extract_amazon_price(soup)
        rating = self._extract_amazon_rating(soup)
        image_url = self._extract_amazon_image(soup)
        
        # Extract ASIN for review scraping
        asin = self._extract_asin(url, soup)
        
        # Scrape reviews
        reviews = []
        if asin:
            reviews = await self._scrape_amazon_reviews(asin)
        
        # Fallback: Try to get reviews from product page if review scraping failed
        if len(reviews) == 0:
            logger.info("No reviews from review pages, trying product page...")
            reviews = self._extract_reviews_from_product_page(soup)
        
        # Extract product features/specifications
        features = self._extract_amazon_features(soup)
        
        return {
            'success': True,
            'platform': 'Amazon',
            'url': url,
            'product_name': product_name,
            'price': price,
            'currency': 'INR' if 'amazon.in' in url else 'USD',
            'rating': rating,
            'image_url': image_url,
            'features': features,
            'reviews': reviews,
            'review_count': len(reviews),
            'asin': asin,
        }
    
    def _extract_amazon_title(self, soup: BeautifulSoup) -> str:
        """Extract Amazon product title"""
        selectors = [
            '#productTitle',
            '#title',
            'span#productTitle',
            'h1.product-title',
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text(strip=True)
        
        return "Unknown Product"
    
    def _extract_amazon_price(self, soup: BeautifulSoup) -> Optional[float]:
        """Extract Amazon product price"""
        selectors = [
            '.a-price-whole',
            '#priceblock_ourprice',
            '#priceblock_dealprice',
            '.a-price .a-offscreen',
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                text = element.get_text(strip=True)
                # Extract numbers only
                match = re.search(r'[\d,]+\.?\d*', text.replace(',', ''))
                if match:
                    return float(match.group().replace(',', ''))
        
        return None
    
    def _extract_amazon_rating(self, soup: BeautifulSoup) -> Optional[float]:
        """Extract Amazon product rating"""
        selectors = [
            'span.a-icon-alt',
            '#acrPopover',
            '.a-star-medium .a-icon-alt',
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                text = element.get_text(strip=True)
                match = re.search(r'(\d+\.?\d*)\s*out of', text)
                if match:
                    return float(match.group(1))
        
        return None
    
    def _extract_amazon_image(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract Amazon product main image"""
        selectors = [
            '#landingImage',
            '#imgBlkFront',
            '#main-image',
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element and element.get('src'):
                return element['src']
        
        return None
    
    def _extract_asin(self, url: str, soup: BeautifulSoup) -> Optional[str]:
        """Extract Amazon ASIN from URL or page"""
        # Try URL patterns - multiple formats
        patterns = [
            r'/dp/([A-Z0-9]{10})',
            r'/product/([A-Z0-9]{10})',
            r'/gp/product/([A-Z0-9]{10})',
            r'ASIN[=/:]([A-Z0-9]{10})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                asin = match.group(1)
                logger.info(f"Extracted ASIN from URL: {asin}")
                return asin
        
        # Try page content
        asin_input = soup.find('input', {'name': 'ASIN'})
        if asin_input:
            asin = asin_input.get('value')
            logger.info(f"Extracted ASIN from page: {asin}")
            return asin
        
        # Try data-asin attribute
        elements_with_asin = soup.find_all(attrs={'data-asin': True})
        for element in elements_with_asin:
            asin = element.get('data-asin')
            if asin and len(asin) == 10:
                logger.info(f"Extracted ASIN from data-asin: {asin}")
                return asin
        
        logger.warning("Could not extract ASIN from URL or page")
        return None
    
    def _extract_amazon_features(self, soup: BeautifulSoup) -> List[str]:
        """Extract product features/bullet points"""
        features = []
        
        # Feature bullets
        feature_div = soup.find('div', {'id': 'feature-bullets'})
        if feature_div:
            for li in feature_div.find_all('li'):
                text = li.get_text(strip=True)
                if text and len(text) > 10:
                    features.append(text)
        
        # Product details table
        detail_table = soup.find('table', {'id': 'productDetails_techSpec_section_1'})
        if detail_table:
            for row in detail_table.find_all('tr'):
                cells = row.find_all('td')
                if len(cells) == 2:
                    key = cells[0].get_text(strip=True)
                    value = cells[1].get_text(strip=True)
                    if key and value:
                        features.append(f"{key}: {value}")
        
        return features[:20]  # Limit to top 20 features
    
    def _extract_reviews_from_product_page(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract reviews directly from product page (fallback)"""
        reviews = []
        
        try:
            # Find review sections on product page
            review_sections = (
                soup.find_all('div', {'data-hook': 'review'}) or
                soup.find_all('div', id=re.compile(r'.*review.*', re.I))
            )
            
            logger.info(f"Found {len(review_sections)} reviews on product page")
            
            for review_div in review_sections[:20]:  # Limit to 20 from product page
                review = self._parse_amazon_review(review_div)
                if review:
                    reviews.append(review)
        
        except Exception as e:
            logger.error(f"Error extracting reviews from product page: {e}")
        
        return reviews
    
    async def _scrape_amazon_reviews(self, asin: str, max_reviews: int = 100) -> List[Dict]:
        """Scrape Amazon product reviews"""
        reviews = []
        
        # Try multiple review URL formats
        base_urls = [
            f"https://www.amazon.in/product-reviews/{asin}/ref=cm_cr_dp_d_show_all_btm",
            f"https://www.amazon.in/product-reviews/{asin}",
            f"https://www.amazon.in/{asin}/product-reviews",
        ]
        
        logger.info(f"Starting review scraping for ASIN: {asin}")
        
        for base_url in base_urls:
            logger.info(f"Trying review URL: {base_url}")
            
            for page in range(1, 6):  # Scrape first 5 pages
                url = f"{base_url}?pageNumber={page}"
                html = await self._fetch_page(url)
                
                if not html:
                    logger.warning(f"Failed to fetch page {page} from {base_url}")
                    break
                
                soup = BeautifulSoup(html, 'lxml')
                
                # Try multiple review selectors
                review_divs = (
                    soup.find_all('div', {'data-hook': 'review'}) or
                    soup.find_all('div', class_='review') or
                    soup.find_all('div', class_=re.compile(r'.*review.*', re.I))
                )
                
                logger.info(f"Found {len(review_divs)} review divs on page {page}")
                
                if not review_divs:
                    if page == 1:
                        # Try next base URL
                        continue
                    else:
                        # No more reviews on this base URL
                        break
                
                for review_div in review_divs:
                    review = self._parse_amazon_review(review_div)
                    if review:
                        reviews.append(review)
                    
                    if len(reviews) >= max_reviews:
                        logger.info(f"Reached max reviews limit: {max_reviews}")
                        return reviews
                
                # If we got reviews from this base URL, continue with it
                if len(reviews) > 0:
                    await asyncio.sleep(1)  # Rate limiting
                else:
                    # Try next base URL
                    break
            
            # If we got reviews, don't try other base URLs
            if len(reviews) > 0:
                break
        
        logger.info(f"Total reviews scraped: {len(reviews)}")
        return reviews
    
    def _parse_amazon_review(self, review_div) -> Optional[Dict]:
        """Parse single Amazon review"""
        try:
            # Rating - try multiple selectors
            rating_element = (
                review_div.find('i', {'data-hook': 'review-star-rating'}) or
                review_div.find('i', class_=re.compile(r'.*star.*', re.I)) or
                review_div.find('span', class_=re.compile(r'.*rating.*', re.I))
            )
            rating = None
            if rating_element:
                rating_text = rating_element.get_text(strip=True)
                match = re.search(r'(\d+\.?\d*)', rating_text)
                if match:
                    rating = float(match.group(1))
            
            # Title - try multiple selectors
            title_element = (
                review_div.find('a', {'data-hook': 'review-title'}) or
                review_div.find('span', {'data-hook': 'review-title'}) or
                review_div.find(class_=re.compile(r'.*review.*title.*', re.I))
            )
            title = title_element.get_text(strip=True) if title_element else ""
            
            # Review text - try multiple selectors
            text_element = (
                review_div.find('span', {'data-hook': 'review-body'}) or
                review_div.find('div', {'data-hook': 'review-body'}) or
                review_div.find(class_=re.compile(r'.*review.*text.*', re.I)) or
                review_div.find(class_=re.compile(r'.*review.*body.*', re.I))
            )
            text = text_element.get_text(strip=True) if text_element else ""
            
            # Reviewer - try multiple selectors
            reviewer_element = (
                review_div.find('span', {'class': 'a-profile-name'}) or
                review_div.find('div', {'class': 'a-profile-name'}) or
                review_div.find(class_=re.compile(r'.*profile.*name.*', re.I))
            )
            reviewer = reviewer_element.get_text(strip=True) if reviewer_element else "Anonymous"
            
            # Date - try multiple selectors
            date_element = (
                review_div.find('span', {'data-hook': 'review-date'}) or
                review_div.find(class_=re.compile(r'.*review.*date.*', re.I))
            )
            date = date_element.get_text(strip=True) if date_element else ""
            
            # Helpful votes
            helpful_element = review_div.find('span', {'data-hook': 'helpful-vote-statement'})
            helpful_count = 0
            if helpful_element:
                helpful_text = helpful_element.get_text(strip=True)
                match = re.search(r'(\d+)', helpful_text)
                if match:
                    helpful_count = int(match.group(1))
            
            # Verified purchase
            verified = review_div.find('span', {'data-hook': 'avp-badge'}) is not None
            
            # Only return if we have meaningful text
            if text and len(text) > 20:
                return {
                    'reviewer': reviewer,
                    'rating': rating,
                    'title': title,
                    'text': text,
                    'date': date,
                    'helpful_count': helpful_count,
                    'verified_purchase': verified,
                }
            else:
                logger.debug(f"Skipping review with insufficient text (length: {len(text)})")
        
        except Exception as e:
            logger.error(f"Error parsing review: {e}")
        
        return None
    
    async def _scrape_flipkart(self, url: str) -> Dict:
        """Scrape Flipkart product details and reviews"""
        html = await self._fetch_page(url)
        if not html:
            return self._empty_result("Failed to fetch Flipkart page")
        
        soup = BeautifulSoup(html, 'lxml')
        
        # Extract product details
        product_name = self._extract_flipkart_title(soup)
        price = self._extract_flipkart_price(soup)
        rating = self._extract_flipkart_rating(soup)
        image_url = self._extract_flipkart_image(soup)
        reviews = self._extract_flipkart_reviews(soup)
        features = self._extract_flipkart_features(soup)
        
        return {
            'success': True,
            'platform': 'Flipkart',
            'url': url,
            'product_name': product_name,
            'price': price,
            'currency': 'INR',
            'rating': rating,
            'image_url': image_url,
            'features': features,
            'reviews': reviews,
            'review_count': len(reviews),
        }
    
    def _extract_flipkart_title(self, soup: BeautifulSoup) -> str:
        """Extract Flipkart product title"""
        selectors = [
            'span.VU-ZEz',
            'span.B_NuCI',
            'h1.yhB1nd',
            '.product-title',
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text(strip=True)
        
        return "Unknown Product"
    
    def _extract_flipkart_price(self, soup: BeautifulSoup) -> Optional[float]:
        """Extract Flipkart product price"""
        selectors = [
            'div.Nx9bqj',
            'div._30jeq3',
            'div._25b18c',
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                text = element.get_text(strip=True)
                match = re.search(r'[\d,]+', text.replace(',', ''))
                if match:
                    return float(match.group())
        
        return None
    
    def _extract_flipkart_rating(self, soup: BeautifulSoup) -> Optional[float]:
        """Extract Flipkart product rating"""
        selectors = [
            'div.XQDdHH',
            'div._3LWZlK',
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                text = element.get_text(strip=True)
                match = re.search(r'(\d+\.?\d*)', text)
                if match:
                    return float(match.group(1))
        
        return None
    
    def _extract_flipkart_image(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract Flipkart product main image"""
        selectors = [
            'img._396cs4',
            'img._2r_T1I',
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element and element.get('src'):
                return element['src']
        
        return None
    
    def _extract_flipkart_features(self, soup: BeautifulSoup) -> List[str]:
        """Extract Flipkart product features"""
        features = []
        
        # Specifications table
        spec_tables = soup.find_all('table', {'class': '_14cfVK'})
        for table in spec_tables:
            for row in table.find_all('tr'):
                cells = row.find_all('td')
                if len(cells) == 2:
                    key = cells[0].get_text(strip=True)
                    value = cells[1].get_text(strip=True)
                    if key and value:
                        features.append(f"{key}: {value}")
        
        return features[:20]
    
    def _extract_flipkart_reviews(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract Flipkart reviews from product page"""
        reviews = []
        review_divs = soup.find_all('div', {'class': 'col _2wzgFH'})
        
        for review_div in review_divs[:50]:  # Limit to 50 reviews
            try:
                # Rating
                rating_div = review_div.find('div', {'class': '_3LWZlK'})
                rating = None
                if rating_div:
                    match = re.search(r'(\d+)', rating_div.get_text(strip=True))
                    if match:
                        rating = float(match.group(1))
                
                # Review text
                text_div = review_div.find('div', {'class': 't-ZTKy'})
                text = text_div.get_text(strip=True) if text_div else ""
                
                # Title
                title_div = review_div.find('p', {'class': '_2-N8zT'})
                title = title_div.get_text(strip=True) if title_div else ""
                
                # Reviewer
                reviewer_div = review_div.find('p', {'class': '_2sc7ZR'})
                reviewer = reviewer_div.get_text(strip=True) if reviewer_div else "Anonymous"
                
                if text and len(text) > 10:
                    reviews.append({
                        'reviewer': reviewer,
                        'rating': rating,
                        'title': title,
                        'text': text,
                        'verified_purchase': True,
                    })
            
            except Exception as e:
                logger.error(f"Error parsing Flipkart review: {e}")
        
        return reviews
    
    async def _scrape_myntra(self, url: str) -> Dict:
        """Scrape Myntra product details and reviews"""
        html = await self._fetch_page(url)
        if not html:
            return self._empty_result("Failed to fetch Myntra page")
        
        soup = BeautifulSoup(html, 'lxml')
        
        # Myntra uses heavy JavaScript, so we get limited data
        product_name = self._extract_myntra_title(soup)
        price = self._extract_myntra_price(soup)
        rating = self._extract_myntra_rating(soup)
        
        return {
            'success': True,
            'platform': 'Myntra',
            'url': url,
            'product_name': product_name,
            'price': price,
            'currency': 'INR',
            'rating': rating,
            'reviews': [],  # Myntra requires API calls or Selenium
            'review_count': 0,
            'note': 'Myntra reviews require advanced scraping with Selenium',
        }
    
    def _extract_myntra_title(self, soup: BeautifulSoup) -> str:
        """Extract Myntra product title"""
        title = soup.find('h1', {'class': 'pdp-title'})
        if title:
            return title.get_text(strip=True)
        return "Unknown Product"
    
    def _extract_myntra_price(self, soup: BeautifulSoup) -> Optional[float]:
        """Extract Myntra product price"""
        price = soup.find('span', {'class': 'pdp-price'})
        if price:
            text = price.get_text(strip=True)
            match = re.search(r'[\d,]+', text.replace(',', ''))
            if match:
                return float(match.group())
        return None
    
    def _extract_myntra_rating(self, soup: BeautifulSoup) -> Optional[float]:
        """Extract Myntra product rating"""
        rating = soup.find('div', {'class': 'index-overallRating'})
        if rating:
            match = re.search(r'(\d+\.?\d*)', rating.get_text(strip=True))
            if match:
                return float(match.group(1))
        return None
    
    async def _scrape_generic(self, url: str) -> Dict:
        """Generic scraper for unknown e-commerce sites"""
        html = await self._fetch_page(url)
        if not html:
            return self._empty_result("Failed to fetch page")
        
        soup = BeautifulSoup(html, 'lxml')
        
        # Try to find product name
        product_name = None
        for selector in ['h1', '.product-title', '#product-title', '.product-name']:
            element = soup.select_one(selector)
            if element:
                product_name = element.get_text(strip=True)
                break
        
        # Try to find price
        price = None
        for selector in ['.price', '.product-price', '#price', '.amount']:
            element = soup.select_one(selector)
            if element:
                text = element.get_text(strip=True)
                match = re.search(r'[\d,]+\.?\d*', text.replace(',', ''))
                if match:
                    price = float(match.group().replace(',', ''))
                    break
        
        return {
            'success': True,
            'platform': 'Generic',
            'url': url,
            'product_name': product_name or "Unknown Product",
            'price': price,
            'currency': 'INR',
            'reviews': [],
            'review_count': 0,
            'note': 'Generic scraper with limited capabilities',
        }
    
    def _empty_result(self, error_message: str) -> Dict:
        """Return empty result with error"""
        return {
            'success': False,
            'error': error_message,
            'platform': 'Unknown',
            'reviews': [],
            'review_count': 0,
        }
    
    async def search_alternative_products(self, product_name: str, platform: str = 'amazon', max_results: int = 5) -> List[Dict]:
        """
        Search for alternative products with similar features
        
        Args:
            product_name: Name of the product to find alternatives for
            platform: E-commerce platform to search on
            max_results: Maximum number of alternatives to return
        
        Returns:
            List of alternative products with basic info
        """
        logger.info(f"[INFO] Searching for alternatives to: {product_name} on {platform}")
        
        alternatives = []
        
        try:
            # Extract product category/keywords from name
            keywords = self._extract_search_keywords(product_name)
            
            if platform.lower() == 'amazon':
                alternatives = await self._search_amazon_alternatives(keywords, max_results)
            elif platform.lower() == 'flipkart':
                alternatives = await self._search_flipkart_alternatives(keywords, max_results)
            else:
                logger.warning(f"[WARNING] Alternative search not implemented for {platform}")
        
        except Exception as e:
            logger.error(f"[ERROR] Failed to search alternatives: {e}")
        
        return alternatives
    
    def _extract_search_keywords(self, product_name: str) -> str:
        """Extract search keywords from product name"""
        # Remove common words and extract important keywords
        stopwords = {'the', 'a', 'an', 'with', 'for', 'and', 'or', 'of', 'in', 'on'}
        words = product_name.lower().split()
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        return ' '.join(keywords[:5])  # Top 5 keywords
    
    async def _search_amazon_alternatives(self, keywords: str, max_results: int) -> List[Dict]:
        """Search Amazon for alternative products"""
        alternatives = []
        
        try:
            # Amazon search URL
            search_url = f"https://www.amazon.in/s?k={keywords.replace(' ', '+')}"
            logger.info(f"[INFO] Searching Amazon: {search_url}")
            
            async with self.session.get(search_url, headers=self._get_headers(), timeout=15) as response:
                logger.info(f"[INFO] Amazon search response status: {response.status}")
                
                if response.status != 200:
                    logger.warning(f"[WARNING] Amazon search failed with status {response.status}")
                    return alternatives
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Find product listings - try multiple selectors
                product_divs = soup.find_all('div', {'data-component-type': 's-search-result'})
                
                if not product_divs:
                    # Fallback: try alternative selectors
                    product_divs = soup.find_all('div', {'data-asin': True})
                    logger.info(f"[INFO] Using fallback selector, found {len(product_divs)} products")
                else:
                    logger.info(f"[INFO] Found {len(product_divs)} products with primary selector")
                
                product_divs = product_divs[:max_results]
                
                for idx, div in enumerate(product_divs):
                    try:
                        # Extract product info
                        name_elem = div.find('h2')
                        if not name_elem:
                            name_elem = div.find('span', {'class': 'a-size-medium'})
                        name = name_elem.get_text(strip=True) if name_elem else None
                        
                        # Get product link
                        link_elem = div.find('a', {'class': 'a-link-normal'})
                        if not link_elem:
                            link_elem = div.find('a', href=True)
                        
                        product_url = None
                        if link_elem and 'href' in link_elem.attrs:
                            href = link_elem['href']
                            if href.startswith('http'):
                                product_url = href
                            else:
                                product_url = f"https://www.amazon.in{href}"
                        
                        # Get rating
                        rating_elem = div.find('span', {'class': 'a-icon-alt'})
                        rating_text = rating_elem.get_text(strip=True) if rating_elem else None
                        rating = None
                        if rating_text:
                            match = re.search(r'(\d+\.?\d*)', rating_text)
                            if match:
                                rating = float(match.group(1))
                        
                        # Get price
                        price_elem = div.find('span', {'class': 'a-price-whole'})
                        price = None
                        if price_elem:
                            price_text = price_elem.get_text(strip=True).replace(',', '').replace('₹', '')
                            match = re.search(r'(\d+)', price_text)
                            if match:
                                price = float(match.group(1))
                        
                        # Get review count
                        review_elem = div.find('span', {'class': 'a-size-base'})
                        review_count = None
                        if review_elem:
                            review_text = review_elem.get_text(strip=True).replace(',', '')
                            match = re.search(r'(\d+)', review_text)
                            if match:
                                review_count = int(match.group(1))
                        
                        # Get image
                        img_elem = div.find('img')
                        image_url = None
                        if img_elem:
                            if 'src' in img_elem.attrs:
                                image_url = img_elem['src']
                            elif 'data-src' in img_elem.attrs:
                                image_url = img_elem['data-src']
                        
                        if name and product_url:
                            logger.info(f"[INFO] Alternative {idx+1}: {name[:50]}...")
                            alternatives.append({
                                'name': name,
                                'url': product_url,
                                'price': price,
                                'rating': rating,
                                'review_count': review_count,
                                'image_url': image_url,
                                'platform': 'Amazon',
                            })
                        else:
                            logger.warning(f"[WARNING] Skipping product {idx+1}: name={bool(name)}, url={bool(product_url)}")
                    
                    except Exception as e:
                        logger.error(f"[ERROR] Failed to parse alternative product {idx+1}: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"[ERROR] Amazon alternative search failed: {e}")
            import traceback
            traceback.print_exc()
        
        return alternatives
    
    async def _search_flipkart_alternatives(self, keywords: str, max_results: int) -> List[Dict]:
        """Search Flipkart for alternative products"""
        alternatives = []
        
        try:
            # Flipkart search URL
            search_url = f"https://www.flipkart.com/search?q={keywords.replace(' ', '+')}"
            
            async with self.session.get(search_url, headers=self._get_headers()) as response:
                if response.status != 200:
                    logger.warning(f"[WARNING] Flipkart search failed with status {response.status}")
                    return alternatives
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Find product listings (Flipkart structure)
                product_divs = soup.find_all('div', {'class': '_1AtVbE'})[:max_results]
                
                for div in product_divs:
                    try:
                        # Extract product info
                        name_elem = div.find('div', class_='_4rR01T')
                        name = name_elem.get_text(strip=True) if name_elem else None
                        
                        # Get product link
                        link_elem = div.find('a', class_='_1fQZEK')
                        product_url = f"https://www.flipkart.com{link_elem['href']}" if link_elem and 'href' in link_elem.attrs else None
                        
                        # Get rating
                        rating_elem = div.find('div', class_='_3LWZlK')
                        rating = float(rating_elem.get_text(strip=True)) if rating_elem else None
                        
                        # Get price
                        price_elem = div.find('div', class_='_30jeq3')
                        price = None
                        if price_elem:
                            price_text = price_elem.get_text(strip=True).replace('₹', '').replace(',', '')
                            match = re.search(r'(\d+)', price_text)
                            if match:
                                price = float(match.group(1))
                        
                        # Get image
                        img_elem = div.find('img', class_='_396cs4')
                        image_url = img_elem['src'] if img_elem and 'src' in img_elem.attrs else None
                        
                        if name and product_url:
                            alternatives.append({
                                'name': name,
                                'url': product_url,
                                'price': price,
                                'rating': rating,
                                'review_count': None,
                                'image_url': image_url,
                                'platform': 'Flipkart',
                            })
                    
                    except Exception as e:
                        logger.error(f"[ERROR] Failed to parse Flipkart alternative: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"[ERROR] Flipkart alternative search failed: {e}")
        
        return alternatives


# Convenience function
async def scrape_product_url(url: str) -> Dict:
    """Convenience function to scrape a single product URL"""
    async with ProductScraper() as scraper:
        return await scraper.scrape_product(url)
