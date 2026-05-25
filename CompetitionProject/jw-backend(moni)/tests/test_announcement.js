const fs = require('fs');
const path = require('path');
const cheerio = require('cheerio');

async function test() {
    const fixturePath = path.resolve(__dirname, 'fixtures', 'test_announcement.html');
    
    try {
        const html = fs.readFileSync(fixturePath, 'utf-8');
        const $ = cheerio.load(html);
        
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
