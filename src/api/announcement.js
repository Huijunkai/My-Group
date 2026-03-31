const cheerio = require('cheerio');
const axios = require('axios');

const ANNOUNCEMENT_URLS = [
    { url: 'https://jwc.bwgl.cn/tzgg/A130008index_1.htm', baseUrl: 'https://jwc.bwgl.cn' },
    { url: 'https://wlxy.bwgl.cn/tzgg/A144010index_1.htm', baseUrl: 'https://wlxy.bwgl.cn' }
];

async function getAnnouncements(limit = 5, offset = 0) {
    try {
        const allAnnouncements = [];

        console.log('开始获取公告，来源:', ANNOUNCEMENT_URLS.map(u => u.url));

        for (const source of ANNOUNCEMENT_URLS) {
            try {
                console.log(`开始获取 ${source.url} 的公告`);
                const response = await axios.get(source.url, {
                    headers: {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8'
                    },
                    timeout: 10000
                });

                const $ = cheerio.load(response.data);
                const announcementsFromSource = [];

                $('.n_right_list1 li a').each((index, element) => {
                    const $link = $(element);
                    const href = $link.attr('href');

                    if (!href) return;

                    const $time = $link.find('.time');
                    const $nr = $link.find('.nr');
                    
                    let date = '';
                    if ($time.length) {
                        const day = $time.find('em').text().trim();
                        const yearMonth = $time.find('i').text().trim();
                        if (day && yearMonth) {
                        const yearShort = yearMonth.substring(2, 4);
                        const month = yearMonth.substring(5);
                        date = `20${yearShort}-${month}-${day.padStart(2, '0')}`;
                    }
                    }

                    const title = $nr.text().trim();
                    if (!title) return;

                    let fullUrl = href;
                    if (href.startsWith('//')) {
                        fullUrl = 'https:' + href;
                    } else if (href.startsWith('/')) {
                        fullUrl = source.baseUrl + href;
                    } else if (href.startsWith('./')) {
                        fullUrl = source.url.substring(0, source.url.lastIndexOf('/') + 1) + href.substring(2);
                    } else if (!href.startsWith('http')) {
                        fullUrl = source.baseUrl + '/' + href;
                    }

                    const announcement = {
                        title: title,
                        url: fullUrl,
                        date: date,
                        id: allAnnouncements.length + 1
                    };
                    
                    announcementsFromSource.push(announcement);
                    allAnnouncements.push(announcement);
                });

                console.log(`从 ${source.url} 获取到 ${announcementsFromSource.length} 条公告`);
            } catch (error) {
                console.error(`获取 ${source.url} 公告失败:`, error.message);
                continue;
            }
        }

        console.log(`总共获取到 ${allAnnouncements.length} 条公告`);

        // 按日期排序，最新的在前
        allAnnouncements.sort((a, b) => {
            return new Date(b.date) - new Date(a.date);
        });

        // 应用分页
        const startIndex = offset;
        const endIndex = startIndex + limit;
        const paginatedAnnouncements = allAnnouncements.slice(startIndex, endIndex);
        
        // 重新分配ID
        paginatedAnnouncements.forEach((announcement, index) => {
            announcement.id = startIndex + index + 1;
        });

        console.log(`最终返回 ${paginatedAnnouncements.length} 条公告，偏移量: ${offset}`);
        return {
            announcements: paginatedAnnouncements,
            total: allAnnouncements.length
        };
    } catch (error) {
        console.error('获取公告失败:', error.message);
        return {
            announcements: [],
            total: 0
        };
    }
}

async function getAnnouncementDetail(url) {
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
        
        let title = '';
        let content = '';
        let date = '';
        const attachments = [];

        const $content = $('#fox_cc');

        const titleMatch = response.data.match(/<title>([^<]*)<\/title>/i);
        if (titleMatch) {
            const fullTitle = titleMatch[1].trim();
            const siteSuffixes = ['通知公告', '教务处', '教务在线', '教学管理', 'News', 'News & Events'];
            let foundSeparator = false;
            
            for (const suffix of siteSuffixes) {
                const pattern = new RegExp(`\\s*[-|]\\s*${suffix}\\s*$`, 'i');
                if (pattern.test(fullTitle)) {
                    title = fullTitle.replace(pattern, '').trim();
                    foundSeparator = true;
                    break;
                }
            }
            
            if (!foundSeparator) {
                title = fullTitle;
            }
        }

        if (!title || title === '通知公告') {
            const $pageTitle = $content.find('.n_new_title').first();
            if ($pageTitle.length) {
                const pageTitleText = $pageTitle.text().trim();
                if (pageTitleText && pageTitleText !== '通知公告') {
                    title = pageTitleText;
                }
            }
        }

        if (!title || title === '通知公告') {
            const $h2 = $content.find('h2').first();
            if ($h2.length) {
                const h2Text = $h2.text().trim();
                if (h2Text && h2Text !== '通知公告') {
                    title = h2Text;
                }
            }
        }

        if ($content.length) {
            $content.find('a[href$=".pdf"], a[href$=".doc"], a[href$=".docx"], a[href$=".xls"], a[href$=".xlsx"], a[href$=".zip"], a[href$=".rar"], a[href$=".7z"], a[href$=".ppt"], a[href$=".pptx"], a[href$=".txt"], a[href$=".wps"]').each((i, el) => {
                const $link = $(el);
                const href = $link.attr('href');
                const linkText = $link.text().trim();
                
                if (href && linkText) {
                    let fullUrl = href;
                    if (href.startsWith('//')) {
                        fullUrl = 'https:' + href;
                    } else if (href.startsWith('/')) {
                        const baseUrl = url.substring(0, url.indexOf('/', 8));
                        fullUrl = baseUrl + href;
                    }
                    
                    attachments.push({
                        name: linkText,
                        url: fullUrl
                    });
                }
            });
            
            $content.find('.n_new_title').remove();
            $content.find('.n_new_info').remove();
            $content.find('.crumbs').remove();
            $content.find('h2').remove();
            $content.find('.info').remove();
            $content.find('script').remove();
            $content.find('style').remove();
            $content.find('.pagelist_yc').remove();
            
            content = $content.html() || '';
        } else {
            content = $('body').html() || '';
        }

        const publishTimeMatch = response.data.match(/发布时间[：:]\s*(\d{4}[-\/年]\d{1,2}[-\/月]\d{1,2}日?)/);
        if (publishTimeMatch) {
            date = publishTimeMatch[1].replace(/[年月]/g, '-').replace('日', '');
        }

        return {
            title,
            content,
            date,
            attachments,
            url
        };
    } catch (error) {
        console.error('获取公告详情失败:', error.message);
        return null;
    }
}

module.exports = {
    getAnnouncements,
    getAnnouncementDetail
};
