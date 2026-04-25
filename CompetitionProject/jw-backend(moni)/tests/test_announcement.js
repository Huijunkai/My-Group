const axios = require('axios');
const cheerio = require('cheerio');

async function test() {
    const url = 'https://jwc.bwgl.cn/tzgg/A130008index_1.htm';
    
    try {
        const response = await axios.get(url, {
            headers: {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8'
            },
            timeout: 10000
        });

        const $ = cheerio.load(response.data);
        
        console.log('=== 页面标题 ===');
        console.log($('title').text());
        
        console.log('\n=== 查找包含 tzgg 的链接 ===');
        $('a').each((i, el) => {
            const href = $(el).attr('href');
            if (href && href.includes('tzgg')) {
                console.log(`[${i}] ${$(el).text().trim()} => ${href}`);
            }
        });
        
        console.log('\n=== 查找列表元素 ===');
        $('ul, li, .list, .news').each((i, el) => {
            const className = $(el).attr('class') || 'no-class';
            const tagName = el.tagName;
            console.log(`[${i}] <${tagName} class="${className}">`);
            if (i < 5) {
                const links = $(el).find('a');
                if (links.length > 0) {
                    links.each((j, link) => {
                        console.log(`    - ${$(link).text().trim()} => ${$(link).attr('href')}`);
                    });
                }
            }
        });
        
        console.log('\n=== 查找表格 ===');
        $('table').each((i, el) => {
            console.log(`[${i}] Table found`);
            $(el).find('a').each((j, link) => {
                console.log(`    - ${$(link).text().trim()} => ${$(link).attr('href')}`);
            });
        });
        
        console.log('\n=== 查找class包含list的元素 ===');
        $('[class*="list"], [class*="news"], [class*="notice"], [class*="article"]').each((i, el) => {
            const className = $(el).attr('class');
            console.log(`[${i}] class="${className}"`);
            $(el).find('a').slice(0, 3).each((j, link) => {
                console.log(`    - ${$(link).text().trim()} => ${$(link).attr('href')}`);
            });
        });

    } catch (error) {
        console.error('Error:', error.message);
    }
}

test();
