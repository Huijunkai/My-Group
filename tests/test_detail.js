const axios = require('axios');
const cheerio = require('cheerio');

async function test() {
    const url = 'https://jwc.bwgl.cn/tzgg/2026/3/926fab7d026342739830104e7dc6a9b1.htm';
    const response = await axios.get(url);
    const $ = cheerio.load(response.data);
    
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
