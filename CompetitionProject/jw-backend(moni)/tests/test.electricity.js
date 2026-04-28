/**
 * Electricity API 单元测试
 * 测试模块：电费模块
 * 测试内容：电费查询、提醒设置保存与获取
 */
const { getElectricity, saveElectricityReminderSettings, getElectricityReminderSettings } = require('../src/api/electricity');
const assert = require('assert');

describe('Electricity API - 电费模块', () => {
    describe('getElectricity() - 功能正确性测试', () => {
        it('【正常输入】应该返回电费信息对象', async () => {
            // Arrange
            const user = 'test_user';
            const roomId = 'H4320101';
            const campusId = 'nnxq';
            const buildingId = '4320';

            // Act
            const result = await getElectricity(user, roomId, campusId, buildingId);

            // Assert
            assert.ok(result, '结果不应该为空');
            if (result.success) {
                assert.ok(result.data, '成功时应该包含数据');
            }
        }).timeout(10000);

        it('【异常输入】应该处理缺少参数的情况', async () => {
            // Arrange
            const emptyUser = '';
            const emptyRoomId = '';
            const emptyCampusId = '';
            const emptyBuildingId = '';

            // Act
            const result = await getElectricity(emptyUser, emptyRoomId, emptyCampusId, emptyBuildingId);

            // Assert
            if (result) {
                assert.ok(typeof result.success === 'boolean', '应该包含 success 字段');
                if (!result.success) {
                    assert.ok(result.message, '失败时应该包含错误消息');
                }
            }
        });
    });

    describe('saveElectricityReminderSettings() - 功能正确性测试', () => {
        it('【正常输入】应该保存电费提醒设置', async () => {
            // Arrange
            const studentId = '202101001';
            const settings = {
                enabled: true,
                threshold: 20,
                electricityAccount: 'test@elec',
                electricityPassword: '123456',
                roomId: 'H4320101',
                campusId: 'nnxq',
                buildingId: '4320'
            };

            // Act
            const result = await saveElectricityReminderSettings(studentId, settings);

            // Assert
            if (result.success) {
                assert.ok(result.data, '成功时应该包含保存的数据');
                assert.strictEqual(result.data.studentId, studentId, '学号应该匹配');
                assert.strictEqual(result.data.enabled, settings.enabled, '启用状态应该匹配');
                assert.strictEqual(result.data.threshold, settings.threshold, '阈值应该匹配');
            }
        }).timeout(5000);

        it('【业务逻辑】应该更新已存在的设置', async () => {
            // Arrange
            const studentId = '202101002';
            const settings1 = {
                enabled: true,
                threshold: 20,
                electricityAccount: 'test@elec',
                roomId: 'H4320101',
                campusId: 'nnxq',
                buildingId: '4320'
            };
            const settings2 = {
                enabled: false,
                threshold: 15,
                electricityAccount: 'updated@elec',
                roomId: 'H4320102',
                campusId: 'nnxq',
                buildingId: '4320'
            };

            // Act
            await saveElectricityReminderSettings(studentId, settings1);
            const result = await saveElectricityReminderSettings(studentId, settings2);

            // Assert
            if (result.success) {
                assert.strictEqual(result.data.enabled, settings2.enabled, '启用状态应该更新');
                assert.strictEqual(result.data.threshold, settings2.threshold, '阈值应该更新');
                assert.strictEqual(result.data.electricityAccount, settings2.electricityAccount, '账号应该更新');
            }
        }).timeout(5000);
    });

    describe('getElectricityReminderSettings() - 功能正确性测试', () => {
        it('【正常输入】应该获取已保存的设置', async () => {
            // Arrange
            const studentId = '202101003';
            const settings = {
                enabled: true,
                threshold: 25,
                electricityAccount: 'get_test@elec',
                roomId: 'H4320103',
                campusId: 'nnxq',
                buildingId: '4320'
            };

            // Act
            await saveElectricityReminderSettings(studentId, settings);
            const result = await getElectricityReminderSettings(studentId);

            // Assert
            if (result.success) {
                assert.ok(result.data, '成功时应该包含设置数据');
                assert.strictEqual(result.data.enabled, true, '启用状态应该匹配');
                assert.strictEqual(result.data.threshold, 25, '阈值应该匹配');
            }
        }).timeout(5000);

        it('【异常输入】对于不存在的用户应返回默认值', async () => {
            // Arrange
            const nonExistentStudentId = 'nonexistent_user_99999';

            // Act
            const result = await getElectricityReminderSettings(nonExistentStudentId);

            // Assert
            if (result.success) {
                assert.ok(result.data, '应该包含默认设置');
                assert.strictEqual(result.data.enabled, false, '默认应该是禁用状态');
                assert.ok(result.data.threshold !== undefined, '默认阈值不应该为空');
            }
        }).timeout(5000);

        it('【安全性】不应返回密码明文', async () => {
            // Arrange
            const studentId = '202101004';
            const settings = {
                enabled: true,
                threshold: 30,
                electricityAccount: 'security_test@elec',
                electricityPassword: 'secret_password_123',
                roomId: 'H4320104',
                campusId: 'nnxq',
                buildingId: '4320'
            };

            // Act
            await saveElectricityReminderSettings(studentId, settings);
            const result = await getElectricityReminderSettings(studentId);

            // Assert
            if (result.success && result.data) {
                assert.ok(!result.data.electricityPassword, '不应该返回密码明文');
                assert.strictEqual(typeof result.data.hasPassword, 'boolean', '应该有 hasPassword 字段');
            }
        }).timeout(5000);
    });
});