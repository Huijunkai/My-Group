/**
 * Auth API 单元测试
 * 测试模块：认证模块
 * 测试内容：登录功能、返回值验证、异常处理、边界测试
 */
const { login } = require('../src/api/auth');
const assert = require('assert');

describe('Auth API - 认证模块', () => {
    describe('login() - 功能正确性测试', () => {
        it('【正常输入】应该成功登录有效的用户凭证', async () => {
            // Arrange - 准备测试数据
            const username = '202101001';
            const password = '123456';

            // Act - 执行测试
            const result = await login(username, password);

            // Assert - 验证结果
            assert.strictEqual(result.success, true, '登录应该成功');
            assert.ok(result.cookies, '应该返回 cookies');
            assert.ok(result.studentInfo, '应该返回学生信息');
            assert.ok(result.nextUrl, '应该返回跳转URL');
        });

        it('【返回值验证】应该返回正确的学生信息结构', async () => {
            // Arrange
            const username = '202101001';
            const password = '123456';

            // Act
            const result = await login(username, password);

            // Assert - 验证返回值结构
            assert.strictEqual(result.studentInfo.studentId, '202101001', '学号应该匹配');
            assert.strictEqual(result.studentInfo.name, '张三', '姓名应该匹配');
            assert.strictEqual(result.studentInfo.gender, '男', '性别应该匹配');
            assert.strictEqual(result.studentInfo.college, '信息工程学院', '学院应该匹配');
            
            // Assert - 验证返回值类型
            assert.strictEqual(typeof result.studentInfo.studentId, 'string', '学号应该是字符串');
            assert.strictEqual(typeof result.studentInfo.name, 'string', '姓名应该是字符串');
            assert.strictEqual(typeof result.studentInfo.gender, 'string', '性别应该是字符串');
            assert.strictEqual(typeof result.studentInfo.college, 'string', '学院应该是字符串');
        });

        it('【返回值验证】cookies 应该包含必要的会话信息', async () => {
            // Arrange
            const username = '202101001';
            const password = '123456';

            // Act
            const result = await login(username, password);

            // Assert
            assert.ok(Array.isArray(result.cookies), 'cookies 应该是数组');
            assert.ok(result.cookies.length >= 1, '至少应该有一个 cookie');
            assert.ok(result.cookies.some(c => c.includes('JSESSIONID')), '应该包含 JSESSIONID');
        });
    });

    describe('login() - 异常处理测试', () => {
        it('【异常输入】应该在密码错误时返回错误消息', async () => {
            // Arrange
            const username = '202101001';
            const wrongPassword = 'wrongpassword';

            // Act
            const result = await login(username, wrongPassword);

            // Assert
            assert.strictEqual(result.success, false, '登录应该失败');
            assert.strictEqual(result.message, '密码错误，请重新输入', '错误消息应该正确');
            assert.ok(!result.cookies, '不应该返回 cookies');
            assert.ok(!result.studentInfo, '不应该返回学生信息');
        });

        it('【异常输入】应该在用户不存在时返回错误消息', async () => {
            // Arrange
            const nonExistentUsername = '99999999';
            const password = '123456';

            // Act
            const result = await login(nonExistentUsername, password);

            // Assert
            assert.strictEqual(result.success, false, '登录应该失败');
            assert.strictEqual(result.message, '该学号不存在', '错误消息应该正确');
        });
    });

    describe('login() - 边界值测试', () => {
        it('【边界值】应该在用户名为空时返回错误', async () => {
            // Arrange
            const emptyUsername = '';
            const password = '123456';

            // Act
            const result = await login(emptyUsername, password);

            // Assert
            assert.strictEqual(result.success, false, '空用户名应该登录失败');
        });

        it('【边界值】应该在密码为空时返回错误', async () => {
            // Arrange
            const username = '202101001';
            const emptyPassword = '';

            // Act
            const result = await login(username, emptyPassword);

            // Assert
            assert.strictEqual(result.success, false, '空密码应该登录失败');
        });

        it('【边界值】应该在用户名和密码都为空时返回错误', async () => {
            // Arrange
            const emptyUsername = '';
            const emptyPassword = '';

            // Act
            const result = await login(emptyUsername, emptyPassword);

            // Assert
            assert.strictEqual(result.success, false, '空用户名和密码应该登录失败');
        });
    });

    describe('login() - 性能测试', () => {
        it('【性能】应该在合理时间内完成登录', async () => {
            // Arrange
            const username = '202101001';
            const password = '123456';
            const maxResponseTime = 2000; // 2秒

            // Act
            const startTime = Date.now();
            await login(username, password);
            const elapsedTime = Date.now() - startTime;

            // Assert
            assert.ok(elapsedTime < maxResponseTime, `响应时间 ${elapsedTime}ms 应该小于 ${maxResponseTime}ms`);
        }).timeout(5000);
    });
});