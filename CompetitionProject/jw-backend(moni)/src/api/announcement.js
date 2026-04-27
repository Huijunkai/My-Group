const { mockAnnouncements, mockAnnouncementDetails } = require('../mockData');

async function getAnnouncements(limit = 5, offset = 0) {
    try {
        console.log('[Mock Announcement] 获取公告列表 - 限制:', limit, '偏移:', offset);
        
        await new Promise(resolve => setTimeout(resolve, 300));
        
        const allAnnouncements = [...mockAnnouncements];
        
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
        
        console.log(`[Mock Announcement] 返回 ${paginatedAnnouncements.length} 条公告，总数: ${allAnnouncements.length}`);
        return {
            announcements: paginatedAnnouncements,
            total: allAnnouncements.length
        };
    } catch (error) {
        console.error('[Mock Announcement] 获取公告失败:', error.message);
        return {
            announcements: [],
            total: 0
        };
    }
}

async function getAnnouncementDetail(url) {
    try {
        console.log('[Mock Announcement] 获取公告详情:', url);
        
        await new Promise(resolve => setTimeout(resolve, 200));
        
        // 从模拟数据中查找对应URL的详情
        const detail = mockAnnouncementDetails.find(d => d.url === url);
        
        if (detail) {
            return detail;
        }
        
        // 如果没有找到，返回默认详情
        return {
            title: '公告详情',
            content: '<p>这是一条模拟公告详情内容。</p><p>包含了公告的详细信息和相关说明。</p>',
            date: new Date().toISOString().split('T')[0],
            attachments: [],
            url: url
        };
    } catch (error) {
        console.error('[Mock Announcement] 获取公告详情失败:', error.message);
        return null;
    }
}

module.exports = {
    getAnnouncements,
    getAnnouncementDetail
};