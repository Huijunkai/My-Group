const fs = require('fs');
const path = require('path');
const cheerio = require('cheerio');

async function test() {
    const htmlPath = path.resolve(__dirname, 'fixtures', 'test_attachment.html');
    const html = fs.readFileSync(htmlPath, 'utf8');
    const $ = cheerio.load(html);
    
    console.log('=== 查找附件 ===');
    $('a[href$=".pdf"], a[href$=".doc"], a[href$=".docx"], a[href$=".xls"], a[href$=".xlsx"]').each((i, el) => {
        console.log(`[${i}] ${$(el).text().trim()} => ${$(el).attr('href')}`);
    });
    
    console.log('\n=== 查找包含"附件"的链接 ===');
    $('a').each((i, el) => {
        const text = $(el).text().trim();
        const href = $(el).attr('href');
        if (text.includes('附件') || text.includes('.pdf') || text.includes('.doc')) {
            console.log(`[${i}] ${text} => ${href}`);
        }
    });
    
    console.log('\n=== #fox_cc 内容区域 ===');
    const $content = $('#fox_cc');
    console.log('Links in #fox_cc:');
    $content.find('a').each((i, el) => {
        console.log(`  [${i}] ${$(el).text().trim().substring(0, 50)} => ${$(el).attr('href')}`);
    });
}

test();
