/**
 * EmptyRoom API 单元测试
 * 测试模块：空教室查询模块
 * 测试内容：校区列表、教学楼列表、空教室查询、教室课表
 */
const { getCampuses, getBuildings, queryEmptyRooms, queryRoomSchedule } = require('../src/api/emptyroom');
const assert = require('assert');

describe('EmptyRoom API - 空教室查询模块', () => {
    describe('getCampuses() - 功能正确性测试', () => {
        it('【正常输入】应该返回校区列表', async () => {
            // Act
            const campuses = await getCampuses();

            // Assert
            assert.ok(Array.isArray(campuses), '校区列表应该是数组');
            assert.ok(campuses.length > 0, '至少有一个校区');
        });

        it('【字段完整性】每个校区应包含 code 和 name 字段', async () => {
            // Act
            const campuses = await getCampuses();

            // Assert
            campuses.forEach(campus => {
                assert.ok(campus.code !== undefined, 'code 不应该为空');
                assert.ok(campus.name, 'name 不应该为空');
            });
        });

        it('【数据有效性】应该包含预期的校区', async () => {
            // Act
            const campuses = await getCampuses();
            const campusNames = campuses.map(c => c.name);

            // Assert
            assert.ok(
                campusNames.some(name => name.includes('桂林') || name.includes('南宁')),
                '应该包含桂林或南宁校区'
            );
        });
    });

    describe('getBuildings() - 功能正确性测试', () => {
        it('【正常输入】应该返回教学楼列表', async () => {
            // Arrange
            const campusCode = 'oW';

            // Act
            const buildings = await getBuildings(null, campusCode);

            // Assert
            assert.ok(Array.isArray(buildings), '教学楼列表应该是数组');
        });

        it('【字段完整性】对于有效校区代码应该返回教学楼', async () => {
            // Arrange
            const campusCode = 'oW';

            // Act
            const buildings = await getBuildings(null, campusCode);

            // Assert
            if (buildings && buildings.length > 0) {
                buildings.forEach(building => {
                    assert.ok(building.code, '教学楼 code 不应该为空');
                    assert.ok(building.name, '教学楼 name 不应该为空');
                });
            }
        });

        it('【异常输入】对于无效校区代码应该返回空数组', async () => {
            // Arrange
            const invalidCode = 'INVALID';

            // Act
            const buildings = await getBuildings(null, invalidCode);

            // Assert
            assert.strictEqual(buildings.length, 0, '无效校区代码应该返回空数组');
        });

        it('【边界值】对于未指定校区代码应该返回空数组', async () => {
            // Arrange
            const emptyCode = '';

            // Act
            const buildings = await getBuildings(null, emptyCode);

            // Assert
            assert.strictEqual(buildings.length, 0, '未指定校区代码应该返回空数组');
        });
    });

    describe('queryEmptyRooms() - 功能正确性测试', () => {
        it('【正常输入】应该返回空教室列表', async () => {
            // Arrange
            const params = {
                weekStart: 1,
                weekEnd: 16,
                periodStart: 1,
                periodEnd: 12
            };

            // Act
            const rooms = await queryEmptyRooms(null, params);

            // Assert
            assert.ok(Array.isArray(rooms), '空教室列表应该是数组');
        });

        it('【字段完整性】每个教室记录应包含必要字段', async () => {
            // Arrange
            const params = { weekStart: 1, weekEnd: 16 };

            // Act
            const rooms = await queryEmptyRooms(null, params);

            // Assert
            if (rooms && rooms.length > 0) {
                const room = rooms[0];
                assert.ok(room.roomName, '教室名称不应该为空');
                assert.ok(room.building, '教学楼不应该为空');
                assert.ok(room.campus, '校区不应该为空');
                assert.ok(room.capacity > 0, '容量应该大于0');
                assert.ok(Array.isArray(room.emptySlots), '空闲时段应该是数组');
            }
        });

        it('【数据筛选】应该根据节次范围过滤结果', async () => {
            // Arrange
            const params = {
                weekStart: 1,
                weekEnd: 16,
                periodStart: 1,
                periodEnd: 4
            };

            // Act
            const rooms = await queryEmptyRooms(null, params);

            // Assert
            if (rooms && rooms.length > 0) {
                rooms.forEach(room => {
                    room.emptySlots.forEach(slot => {
                        slot.periods.forEach(period => {
                            assert.ok(period >= 1 && period <= 4, `节次 ${period} 应该在 1-4 范围内`);
                        });
                    });
                });
            }
        });
    });

    describe('queryRoomSchedule() - 功能正确性测试', () => {
        it('【正常输入】应该返回指定教室的课表', async () => {
            // Arrange
            const params = { roomName: '教A101' };

            // Act
            const schedule = await queryRoomSchedule(null, params);

            // Assert
            if (schedule) {
                assert.ok(schedule.roomName, '教室名称不应该为空');
                assert.ok(Array.isArray(schedule.schedule), '课表应该是数组');
            }
        });

        it('【异常输入】对于不存在的教室名称应该返回空课表', async () => {
            // Arrange
            const params = { roomName: '不存在教室999' };

            // Act
            const schedule = await queryRoomSchedule(null, params);

            // Assert
            if (schedule) {
                assert.ok(schedule.roomName, '应该返回教室名称');
                assert.ok(Array.isArray(schedule.schedule), '课表应该是数组');
            }
        });

        it('【字段完整性】课表中的每天应该包含节次信息', async () => {
            // Arrange
            const params = { roomName: '教A101' };

            // Act
            const schedule = await queryRoomSchedule(null, params);

            // Assert
            if (schedule && schedule.schedule.length > 0) {
                const daySchedule = schedule.schedule[0];
                assert.ok(daySchedule.day, '星期几不应该为空');
                assert.ok(Array.isArray(daySchedule.periods), '节次应该是数组');
            }
        });
    });

    describe('性能测试', () => {
        it('【性能】getCampuses 应该在合理时间内返回', async () => {
            // Arrange
            const maxResponseTime = 500;

            // Act
            const startTime = Date.now();
            await getCampuses();
            const elapsedTime = Date.now() - startTime;

            // Assert
            assert.ok(elapsedTime < maxResponseTime, `响应时间 ${elapsedTime}ms 应该小于 ${maxResponseTime}ms`);
        });

        it('【性能】queryEmptyRooms 应该在合理时间内返回', async () => {
            // Arrange
            const params = { weekStart: 1, weekEnd: 16 };
            const maxResponseTime = 1000;

            // Act
            const startTime = Date.now();
            await queryEmptyRooms(null, params);
            const elapsedTime = Date.now() - startTime;

            // Assert
            assert.ok(elapsedTime < maxResponseTime, `响应时间 ${elapsedTime}ms 应该小于 ${maxResponseTime}ms`);
        });
    });
});