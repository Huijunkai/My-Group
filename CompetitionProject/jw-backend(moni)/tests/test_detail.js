const fs = require('fs');
const path = require('path');
const cheerio = require('cheerio');

async function test() {
    const htmlPath = path.resolve(__dirname, 'fixtures', 'test_detail.html');
    const html = fs.readFileSync(htmlPath, 'utf8');
    const $ = cheerio.load(html);
    
    console.log('=== 查找标题 ===');
    console.log('h1:', $('h1').first().text().trim());
    console.log('.n_new_title h1:', $('.n_new_title h1').first().text().trim());
    console.log('#fox_cc h1:', $('#fox_cc h1').first().text().trim());
    
    console.log('\n=== 查找内容区域 ===');
    console.log('.v_news_content length:', $('.v_news_content').html()?.length);
    console.log('#fox_cc length:', $('#fox_cc').html()?.length);
    
    console.log('\n=== 页面标题 ===');
    console.log('title:', $('title').text());
}

test();
