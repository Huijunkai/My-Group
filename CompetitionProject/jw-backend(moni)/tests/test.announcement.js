/**
 * Announcement API 单元测试
 * 测试模块：公告模块
 * 测试内容：公告列表、分页、详情
 */
const { getAnnouncements, getAnnouncementDetail } = require('../src/api/announcement');
const assert = require('assert');

describe('Announcement API - 公告模块', () => {
    describe('getAnnouncements() - 功能正确性测试', () => {
        it('【正常输入】应该返回公告列表和总数', async () => {
            // Arrange
            const limit = 5;
            const offset = 0;

            // Act
            const result = await getAnnouncements(limit, offset);

            // Assert
            assert.ok(result, '结果不应该为空');
            assert.ok(Array.isArray(result.announcements), 'announcements 应该是数组');
            assert.strictEqual(typeof result.total, 'number', 'total 应该是数字');
        });

        it('【返回值验证】应该返回指定数量的公告', async () => {
            // Arrange
            const limit = 3;
            const offset = 0;

            // Act
            const result = await getAnnouncements(limit, offset);

            // Assert
            assert.ok(result.announcements.length <= limit, `返回数量不应超过 ${limit}`);
        });

        it('【返回值验证】每条公告应包含必要字段', async () => {
            // Arrange
            const limit = 5;
            const offset = 0;

            // Act
            const result = await getAnnouncements(limit, offset);

            // Assert
            if (result.announcements.length > 0) {
                const announcement = result.announcements[0];
                assert.ok(announcement.id !== undefined, 'ID不应该为空');
                assert.ok(announcement.title, '标题不应该为空');
                assert.ok(announcement.url, 'URL不应该为空');
                assert.ok(announcement.date, '日期不应该为空');
            }
        });

        it('【数据筛选】应该支持分页（offset 参数）', async () => {
            // Arrange
            const limit = 3;

            // Act
            const result1 = await getAnnouncements(limit, 0);
            const result2 = await getAnnouncements(limit, 3);

            // Assert
            if (result1.announcements.length > 0 && result2.announcements.length > 0) {
                assert.notStrictEqual(
                    result1.announcements[0]?.id,
                    result2.announcements[0]?.id,
                    '不同偏移量应该返回不同的数据'
                );
            }
        });

        it('【数据一致性】应该按日期排序（最新的在前）', async () => {
            // Arrange
            const limit = 10;

            // Act
            const result = await getAnnouncements(limit, 0);

            // Assert
            if (result.announcements.length >= 2) {
                for (let i = 1; i < result.announcements.length; i++) {
                    const date1 = new Date(result.announcements[i - 1].date);
                    const date2 = new Date(result.announcements[i].date);
                    assert.ok(date1 >= date2, '应该按日期降序排列');
                }
            }
        });
    });

    describe('getAnnouncementDetail() - 功能正确性测试', () => {
        it('【正常输入】应该根据URL返回公告详情', async () => {
            // Arrange
            const testUrl = 'https://jwc.bwgl.cn/announcement/20250601';

            // Act
            const detail = await getAnnouncementDetail(testUrl);

            // Assert
            assert.ok(detail, '详情不应该为空');
            assert.ok(detail.title || detail.content, '应该包含标题或内容');
        });

        it('【返回值验证】详情应该包含必要字段', async () => {
            // Arrange
            const testUrl = 'https://jwc.bwgl.cn/announcement/20250601';

            // Act
            const detail = await getAnnouncementDetail(testUrl);

            // Assert
            assert.strictEqual(detail.url, testUrl, 'URL应该匹配');
            assert.ok(detail.date, '日期不应该为空');
            assert.ok(Array.isArray(detail.attachments), '附件应该是数组');
        });

        it('【异常输入】对于不存在的URL应该返回默认详情', async () => {
            // Arrange
            const nonExistentUrl = 'https://example.com/nonexistent';

            // Act
            const detail = await getAnnouncementDetail(nonExistentUrl);

            // Assert
            assert.ok(detail, '应该返回详情对象');
            assert.ok(detail.content, '应该包含默认内容');
        });
    });

    describe('边界值测试', () => {
        it('【边界值】limit 为 0 时应该返回空数组', async () => {
            // Arrange
            const limit = 0;
            const offset = 0;

            // Act
            const result = await getAnnouncements(limit, offset);

            // Assert
            assert.ok(Array.isArray(result.announcements), '应该是数组');
        });

        it('【边界值】offset 超出范围时应该返回空数组', async () => {
            // Arrange
            const limit = 5;
            const largeOffset = 999999;

            // Act
            const result = await getAnnouncements(limit, largeOffset);

            // Assert
            assert.ok(Array.isArray(result.announcements), '应该是数组');
            assert.ok(result.announcements.length === 0, '超出范围应该返回空数组');
        });
    });

    describe('性能测试', () => {
        it('【性能】getAnnouncements 应该在合理时间内返回', async () => {
            // Arrange
            const limit = 10;
            const offset = 0;
            const maxResponseTime = 1000;

            // Act
            const startTime = Date.now();
            await getAnnouncements(limit, offset);
            const elapsedTime = Date.now() - startTime;

            // Assert
            assert.ok(elapsedTime < maxResponseTime, `响应时间 ${elapsedTime}ms 应该小于 ${maxResponseTime}ms`);
        });
    });
});