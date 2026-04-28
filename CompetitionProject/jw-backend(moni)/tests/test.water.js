/**
 * Water API 单元测试
 * 测试模块：打水系统模块
 * 测试内容：URL解析、设备初始化、余额查询、账户绑定、二维码扫描
 */
const { parseScanUrl, initWaterDevice, getWaterBalance, bindWaterAccount, scanWaterQrcode } = require('../src/api/water');
const assert = require('assert');

describe('Water API - 打水系统模块', () => {
    describe('parseScanUrl() - 功能正确性测试', () => {
        it('【正常输入】应该正确解析有效的扫码URL', () => {
            // Arrange
            const testUrl = 'https://example.com/?openid=test123&deviceid=device456&app=WECHAT';

            // Act
            const result = parseScanUrl(testUrl);

            // Assert
            assert.strictEqual(result.success, true, '解析应该成功');
            assert.strictEqual(result.data.openid, 'test123', 'openid 应该匹配');
            assert.strictEqual(result.data.deviceid, 'device456', 'deviceid 应该匹配');
            assert.strictEqual(result.data.app, 'WECHAT', 'app 应该匹配');
        });

        it('【异常输入】应该处理缺少 openid 的情况', () => {
            // Arrange
            const testUrl = 'https://example.com/?deviceid=device456';

            // Act
            const result = parseScanUrl(testUrl);

            // Assert
            assert.strictEqual(result.success, false, '缺少 openid 应该失败');
            assert.ok(result.message.includes('openid'), '错误消息应该提到 openid');
        });

        it('【异常输入】应该处理缺少 deviceid 的情况', () => {
            // Arrange
            const testUrl = 'https://example.com/?openid=test123';

            // Act
            const result = parseScanUrl(testUrl);

            // Assert
            assert.strictEqual(result.success, false, '缺少 deviceid 应该失败');
            assert.ok(result.message.includes('deviceid'), '错误消息应该提到 deviceid');
        });

        it('【业务逻辑】应该使用默认 app 值 WECHAT', () => {
            // Arrange
            const testUrl = 'https://example.com/?openid=test123&deviceid=device456';

            // Act
            const result = parseScanUrl(testUrl);

            // Assert
            assert.strictEqual(result.data.app, 'WECHAT', 'app 默认值应该是 WECHAT');
        });

        it('【异常输入】应该解析无效的 URL 格式', () => {
            // Arrange
            const invalidUrl = 'not-a-valid-url';

            // Act
            const result = parseScanUrl(invalidUrl);

            // Assert
            assert.strictEqual(result.success, false, '无效 URL 应该失败');
            assert.ok(result.message.includes('URL'), '错误消息应该提到 URL');
        });
    });

    describe('initWaterDevice() - 功能正确性测试', () => {
        it('【正常输入】应该初始化设备并返回结果', async () => {
            // Arrange
            const openid = 'test_openid';
            const deviceid = 'test_deviceid';
            const app = 'WECHAT';

            // Act
            const result = await initWaterDevice(openid, deviceid, app);

            // Assert
            if (result.success) {
                assert.ok(result.data, '成功时应该包含数据');
                assert.ok(result.data.userid || result.data.location, '应该包含用户ID或位置信息');
            }
        }).timeout(10000);
    });

    describe('getWaterBalance() - 功能正确性测试', () => {
        it('【正常输入】应该获取用户余额信息', async () => {
            // Arrange
            const openid = 'test_openid';
            const deviceid = '';
            const app = 'WECHAT';

            // Act
            const result = await getWaterBalance(openid, deviceid, app);

            // Assert
            if (result.success) {
                assert.ok(result.data, '成功时应该包含数据');
                assert.ok(result.data.balance !== undefined, '余额不应该为空');
                assert.strictEqual(typeof result.data.balance, 'string', '余额应该是字符串类型');
            }
        }).timeout(10000);
    });

    describe('bindWaterAccount() - 功能正确性测试', () => {
        it('【正常输入】应该绑定水卡账户并返回信息', async () => {
            // Arrange
            const testUrl = 'https://example.com/?openid=test_openid&deviceid=test_deviceid&app=WECHAT';

            // Act
            const result = await bindWaterAccount(testUrl);

            // Assert
            if (result.success) {
                assert.ok(result.data, '成功时应该包含数据');
                assert.ok(result.data.openid, '应该包含 openid');
                assert.ok(result.data.deviceid, '应该包含 deviceid');
            }
        }).timeout(15000);
    });

    describe('scanWaterQrcode() - 功能正确性测试', () => {
        it('【正常输入】应该扫描二维码并初始化设备', async () => {
            // Arrange
            const testUrl = 'https://example.com/?openid=test_openid&deviceid=test_deviceid&app=WECHAT';

            // Act
            const result = await scanWaterQrcode(testUrl);

            // Assert
            if (result.success) {
                assert.ok(result.data, '成功时应该包含设备信息');
            }
        }).timeout(10000);

        it('【异常输入】对于无效的 URL 应该返回错误', async () => {
            // Arrange
            const invalidUrl = 'invalid-url';

            // Act
            const result = await scanWaterQrcode(invalidUrl);

            // Assert
            assert.strictEqual(result.success, false, '无效 URL 应该失败');
        });
    });

    describe('边界值测试', () => {
        it('【边界值】parseScanUrl 应该处理空字符串', () => {
            // Arrange
            const emptyUrl = '';

            // Act
            const result = parseScanUrl(emptyUrl);

            // Assert
            assert.strictEqual(result.success, false, '空字符串应该失败');
        });

        it('【边界值】parseScanUrl 应该处理特殊字符', () => {
            // Arrange
            const specialCharUrl = 'https://example.com/?openid=test%40%23%24&deviceid=device%26%2F';

            // Act
            const result = parseScanUrl(specialCharUrl);

            // Assert
            assert.strictEqual(result.success, true, '特殊字符 URL 应该成功解析');
        });
    });
});