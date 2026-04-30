/**
 * MockData 单元测试
 * 测试模块：模拟数据模块
 * 测试内容：数据结构、字段完整性、数据有效性、数据一致性
 */
const {
    mockStudents,
    mockTimetable,
    mockGrades,
    mockExams,
    mockPlans,
    mockProgress,
    mockDormitoryBuildings,
    mockAnnouncements,
    getMockElectricity,
    getMockElectricityReminderSettings,
    getMockEmptyRooms
} = require('../src/mockData');
const assert = require('assert');

describe('MockData - 模拟数据模块', () => {
    describe('mockStudents - 学生数据测试', () => {
        it('【数据结构】应该是数组且不为空', () => {
            // Assert
            assert.ok(Array.isArray(mockStudents), '学生列表应该是数组');
            assert.ok(mockStudents.length > 0, '学生列表不应为空');
        });

        it('【字段完整性】每个学生应包含必要字段', () => {
            // Arrange
            const student = mockStudents[0];

            // Assert
            assert.ok(student.studentId, '学号不应该为空');
            assert.ok(student.password, '密码不应该为空');
            assert.ok(student.name, '姓名不应该为空');
            assert.ok(student.gender, '性别不应该为空');
            assert.ok(student.enrollmentYear, '入学年份不应该为空');
            assert.ok(student.className, '班级不应该为空');
            assert.ok(student.major, '专业不应该为空');
            assert.ok(student.college, '学院不应该为空');
        });

        it('【数据一致性】学号应该唯一', () => {
            // Arrange
            const studentIds = mockStudents.map(s => s.studentId);
            const uniqueIds = new Set(studentIds);

            // Assert
            assert.strictEqual(uniqueIds.size, studentIds.length, '学号应该唯一');
        });

        it('【数据有效性】性别值应该是有效的', () => {
            // Assert
            mockStudents.forEach(student => {
                assert.ok(['男', '女'].includes(student.gender), `性别 "${student.gender}" 应该是男或女`);
            });
        });

        it('【数据有效性】入学年份应该是有效的', () => {
            // Assert
            mockStudents.forEach(student => {
                const year = parseInt(student.enrollmentYear);
                assert.ok(!isNaN(year), `入学年份 "${student.enrollmentYear}" 应该是有效数字`);
                assert.ok(year >= 2000 && year <= 2030, `入学年份 ${year} 应该在合理范围内`);
            });
        });
    });

    describe('mockTimetable - 课表数据测试', () => {
        it('【数据结构】应该是数组且不为空', () => {
            // Assert
            assert.ok(Array.isArray(mockTimetable), '课表应该是数组');
            assert.ok(mockTimetable.length > 0, '课表不应为空');
        });

        it('【字段完整性】每条课程记录应包含必要字段', () => {
            // Arrange
            const course = mockTimetable[0];

            // Assert
            assert.ok(course.name, '课程名称不应该为空');
            assert.ok(course.dayOfWeek, '星期几不应该为空');
            assert.ok(course.period, '节次不应该为空');
            assert.ok(course.teacher, '教师不应该为空');
            assert.ok(course.location, '地点不应该为空');
            assert.ok(course.courseType, '课程类型不应该为空');
        });

        it('【数据有效性】星期几的值应该在有效范围内', () => {
            // Arrange
            const validDays = ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日'];

            // Assert
            mockTimetable.forEach(course => {
                assert.ok(validDays.includes(course.dayOfWeek), `${course.dayOfWeek} 不是有效的星期几`);
            });
        });

        it('【数据有效性】节次格式应该正确', () => {
            // Arrange
            const validPeriods = ['01-02', '03-04', '05-06', '07-08', '09-10', '11-12'];

            // Assert
            mockTimetable.forEach(course => {
                assert.ok(validPeriods.includes(course.period), `${course.period} 不是有效的节次格式`);
            });
        });

        it('【数据有效性】课程类型应该是有效的', () => {
            // Arrange
            const validTypes = ['必修课', '选修课', '实践课'];

            // Assert
            mockTimetable.forEach(course => {
                assert.ok(validTypes.includes(course.courseType), `${course.courseType} 不是有效的课程类型`);
            });
        });
    });

    describe('mockGrades - 成绩数据测试', () => {
        it('【数据结构】应该是按学期分组的对象', () => {
            // Assert
            assert.strictEqual(typeof mockGrades, 'object', '成绩应该是对象');
            assert.ok(Object.keys(mockGrades).length > 0, '至少有一个学期的成绩');
        });

        it('【字段完整性】每个学期应该包含成绩数组', () => {
            // Assert
            Object.values(mockGrades).forEach(semesterGrades => {
                assert.ok(Array.isArray(semesterGrades), '学期成绩应该是数组');
            });
        });

        it('【字段完整性】成绩应该包含必要字段', () => {
            // Arrange
            const firstSemester = Object.values(mockGrades)[0];

            // Assert
            if (firstSemester && firstSemester.length > 0) {
                const grade = firstSemester[0];
                assert.ok(grade.courseName || grade.name, '课程名称不应该为空');
                assert.ok(grade.score !== undefined, '分数不应该为空');
                assert.ok(grade.credit !== undefined, '学分不应该为空');
            }
        });
    });

    describe('mockExams - 考试安排测试', () => {
        it('【数据结构】应该是数组', () => {
            // Assert
            assert.ok(Array.isArray(mockExams), '考试安排应该是数组');
        });

        it('【字段完整性】每条考试记录应包含必要字段', () => {
            // Assert
            if (mockExams && mockExams.length > 0) {
                const exam = mockExams[0];
                assert.ok(exam.courseName || exam.name, '课程名称不应该为空');
                assert.ok(exam.examTime || exam.time, '考试时间不应该为空');
                assert.ok(exam.location || exam.place, '考试地点不应该为空');
            }
        });
    });

    describe('mockPlans - 培养计划测试', () => {
        it('【数据结构】应该是按学期分组的对象', () => {
            // Assert
            assert.strictEqual(typeof mockPlans, 'object', '培养计划应该是对象');
        });

        it('【字段完整性】每个计划应该包含学分信息', () => {
            // Arrange
            const firstSemester = Object.values(mockPlans)[0];

            // Assert
            if (firstSemester && firstSemester.length > 0) {
                const plan = firstSemester[0];
                assert.ok(plan.credit !== undefined || plan.totalHours !== undefined, '应该包含学分或学时信息');
            }
        });
    });

    describe('mockProgress - 学分进度测试', () => {
        it('【数据结构】应该是数组', () => {
            // Assert
            assert.ok(Array.isArray(mockProgress), '进度应该是数组');
        });

        it('【字段完整性】每条进度记录应包含类别和学分信息', () => {
            // Assert
            if (mockProgress && mockProgress.length > 0) {
                const item = mockProgress[0];
                assert.ok(item.category, '类别不应该为空');
                assert.ok(item.requiredCredits !== undefined, '要求学分不应该为空');
                assert.ok(item.completedCredits !== undefined, '已完成学分不应该为空');
            }
        });

        it('【数据一致性】已完成学分应该是有效的数值', () => {
            // Assert
            mockProgress.forEach(item => {
                if (item.requiredCredits !== undefined && item.completedCredits !== undefined) {
                    const completed = parseFloat(item.completedCredits);
                    const required = parseFloat(item.requiredCredits);
                    assert.ok(!isNaN(completed), `类别 "${item.category}" 已完成学分应该是有效数值`);
                    assert.ok(!isNaN(required), `类别 "${item.category}" 要求学分应该是有效数值`);
                    assert.ok(completed >= 0, `类别 "${item.category}" 已完成学分应该大于等于0`);
                    assert.ok(required >= 0, `类别 "${item.category}" 要求学分应该大于等于0`);
                }
            });
        });
    });

    describe('mockDormitoryBuildings - 宿舍楼测试', () => {
        it('【数据结构】应该是数组', () => {
            // Assert
            assert.ok(Array.isArray(mockDormitoryBuildings), '宿舍楼列表应该是数组');
        });

        it('【字段完整性】每个宿舍楼应包含楼栋ID和名称', () => {
            // Assert
            if (mockDormitoryBuildings && mockDormitoryBuildings.length > 0) {
                const building = mockDormitoryBuildings[0];
                assert.ok(building.loudong_id, '楼栋ID不应该为空');
                assert.ok(building.loudong_name, '楼栋名称不应该为空');
                assert.ok(building.xiaoqu_id, '校区ID不应该为空');
                assert.ok(building.xiaoqu_name, '校区名称不应该为空');
            }
        });
    });

    describe('mockAnnouncements - 公告测试', () => {
        it('【数据结构】应该是数组', () => {
            // Assert
            assert.ok(Array.isArray(mockAnnouncements), '公告列表应该是数组');
        });

        it('【字段完整性】每条公告应包含标题、URL和日期', () => {
            // Assert
            if (mockAnnouncements && mockAnnouncements.length > 0) {
                const announcement = mockAnnouncements[0];
                assert.ok(announcement.title, '标题不应该为空');
                assert.ok(announcement.url, 'URL不应该为空');
                assert.ok(announcement.date, '日期不应该为空');
            }
        });
    });

    describe('getMockElectricity() - 电费信息测试', () => {
        it('【正常输入】应该返回电费信息对象', () => {
            // Arrange
            const roomId = 'H4320101';

            // Act
            const electricity = getMockElectricity(roomId);

            // Assert
            assert.ok(electricity, '电费信息不应该为空');
            assert.strictEqual(electricity.room_id, roomId, '房间ID应该匹配');
            assert.ok(electricity.balance !== undefined, '余额不应该为空');
            assert.strictEqual(electricity.unit, '元', '单位应该是元');
        });

        it('【数据有效性】余额应该是有效的数值', () => {
            // Arrange
            const roomId = 'H4320101';

            // Act
            const electricity = getMockElectricity(roomId);

            // Assert
            const balance = parseFloat(electricity.balance);
            assert.ok(!isNaN(balance), '余额应该是有效数值');
            assert.ok(balance >= 0, '余额应该大于等于0');
        });
    });

    describe('getMockElectricityReminderSettings() - 电费提醒设置测试', () => {
        it('【正常输入】应该返回电费提醒设置对象', () => {
            // Arrange
            const studentId = '202101001';

            // Act
            const settings = getMockElectricityReminderSettings(studentId);

            // Assert
            assert.ok(settings, '设置不应该为空');
            assert.strictEqual(settings.studentId, studentId, '学号应该匹配');
            assert.ok(settings.enabled !== undefined, '启用状态不应该为空');
            assert.ok(settings.threshold !== undefined, '阈值不应该为空');
            assert.ok(settings.roomId, '房间ID不应该为空');
        });

        it('【数据有效性】阈值应该在合理范围内', () => {
            // Arrange
            const studentId = '202101001';

            // Act
            const settings = getMockElectricityReminderSettings(studentId);

            // Assert
            assert.ok(settings.threshold > 0, '阈值应该大于0');
            assert.ok(settings.threshold <= 100, '阈值应该小于等于100');
        });
    });

    describe('getMockEmptyRooms() - 空教室测试', () => {
        it('【正常输入】应该返回指定星期的空教室数组', () => {
            // Arrange
            const dayOfWeek = '星期一';

            // Act
            const rooms = getMockEmptyRooms(dayOfWeek);

            // Assert
            assert.ok(Array.isArray(rooms), '空教室列表应该是数组');
        });

        it('【字段完整性】每个空教室应包含房间名和空闲节次', () => {
            // Arrange
            const dayOfWeek = '星期一';

            // Act
            const rooms = getMockEmptyRooms(dayOfWeek);

            // Assert
            if (rooms && rooms.length > 0) {
                const room = rooms[0];
                assert.ok(room.room, '房间名不应该为空');
                assert.ok(Array.isArray(room.periods), '空闲节次应该是数组');
            }
        });

        it('【异常输入】对于无效的星期应该返回空数组', () => {
            // Arrange
            const invalidDay = '无效星期';

            // Act
            const rooms = getMockEmptyRooms(invalidDay);

            // Assert
            assert.ok(Array.isArray(rooms), '应该是数组');
            assert.strictEqual(rooms.length, 0, '无效星期应该返回空数组');
        });
    });
});