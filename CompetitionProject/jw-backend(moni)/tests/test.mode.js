/**
 * Mode 单元测试
 * 测试模块：模式配置模块
 * 测试内容：模式判断、返回值验证、数据一致性
 */
const { isMockMode, getModeInfo } = require('../src/mode');
const assert = require('assert');

describe('Mode - 模式配置模块', () => {
    describe('isMockMode() - 功能正确性测试', () => {
        it('【返回值验证】应该返回布尔值', () => {
            // Act
            const result = isMockMode();

            // Assert
            assert.strictEqual(typeof result, 'boolean', '应该返回布尔值');
        });

        it('【业务逻辑】在开发环境下应该返回 true', () => {
            // Arrange
            const isDev = process.env.NODE_ENV === 'development' || process.env.MOCK_MODE === 'true';

            // Act
            const result = isMockMode();

            // Assert
            if (isDev) {
                assert.strictEqual(result, true, '开发环境应该启用模拟模式');
            }
        });
    });

    describe('getModeInfo() - 功能正确性测试', () => {
        it('【返回值验证】应该返回模式信息对象', () => {
            // Act
            const info = getModeInfo();

            // Assert
            assert.ok(info, '模式信息不应该为空');
            assert.strictEqual(typeof info, 'object', '应该是对象');
        });

        it('【字段完整性】应该包含必要字段', () => {
            // Act
            const info = getModeInfo();

            // Assert
            assert.ok(info.mode !== undefined, 'mode 字段不应该为空');
            assert.ok(info.isMock !== undefined, 'isMock 字段不应该为空');
            assert.ok(info.timestamp, 'timestamp 字段不应该为空');
        });

        it('【数据有效性】mode 值应该在有效范围内', () => {
            // Arrange
            const validModes = ['mock-data', 'production'];

            // Act
            const info = getModeInfo();

            // Assert
            assert.ok(
                validModes.includes(info.mode),
                `mode 应该是 "mock-data" 或 "production"，实际是 "${info.mode}"`
            );
        });

        it('【数据一致性】isMock 和 mode 应该一致', () => {
            // Act
            const info = getModeInfo();
            const mockStatus = isMockMode();

            // Assert
            assert.strictEqual(info.isMock, mockStatus, 'isMock 应该与 isMockMode() 返回值一致');
        });

        it('【数据有效性】timestamp 应该是有效的 ISO 日期字符串', () => {
            // Act
            const info = getModeInfo();

            // Assert
            const date = new Date(info.timestamp);
            assert.ok(!isNaN(date.getTime()), 'timestamp 应该是有效的日期');
        });

        it('【数据一致性】timestamp 应该是近期时间', () => {
            // Arrange
            const oneHourAgo = Date.now() - 60 * 60 * 1000;

            // Act
            const info = getModeInfo();
            const timestamp = new Date(info.timestamp).getTime();

            // Assert
            assert.ok(timestamp > oneHourAgo, 'timestamp 应该是近期时间');
        });
    });

    describe('数据一致性测试', () => {
        it('【数据一致性】多次调用 isMockMode 应该返回相同结果', () => {
            // Act
            const result1 = isMockMode();
            const result2 = isMockMode();
            const result3 = isMockMode();

            // Assert
            assert.strictEqual(result1, result2, '多次调用结果应该一致');
            assert.strictEqual(result2, result3, '多次调用结果应该一致');
        });

        it('【数据一致性】多次调用 getModeInfo 应该返回一致的 mode', () => {
            // Act
            const info1 = getModeInfo();
            const info2 = getModeInfo();

            // Assert
            assert.strictEqual(info1.mode, info2.mode, 'mode 应该一致');
            assert.strictEqual(info1.isMock, info2.isMock, 'isMock 应该一致');
        });
    });
});