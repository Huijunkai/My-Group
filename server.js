const express = require('express');
const cors = require('cors');
const { login } = require('./src/api/auth');
const { getStudentInfo, getTimetable, getGrades, getExamSchedule, getSemesterPlan, getStudyProgress } = require('./src/api/student');
const { syncStudent, syncCourses, syncGrades, syncExams, syncPlans, syncProgress } = require('./src/db/sync');
const { Student, Course, Grade, Exam, Plan, Progress } = require('./src/db/models');

const { initDatabase } = require('./src/db');

const app = express();
app.use(cors());
app.use(express.json());

// 端口配置，Railway 会自动注入 PORT 环境变量
const PORT = process.env.PORT || 3000;

// 初始化数据库
let dbReady = false;
initDatabase().then(() => {
    console.log('Database initialized');
    dbReady = true;
}).catch((error) => {
    console.error('Database initialization failed:', error);
    // 即使数据库初始化失败，服务器仍然启动，但会在 API 调用时返回错误
});

/**
 * 根路由测试
 */
app.get('/', (req, res) => {
    res.json({ message: '教务系统同步服务已启动', status: 'running' });
});

/**
 * 版本/部署信息：用于确认 Railway 是否已部署到最新代码
 * GET /api/version
 */
app.get('/api/version', (_req, res) => {
    let pkgVersion = 'unknown';
    try {
        // eslint-disable-next-line global-require
        const pkg = require('./package.json');
        pkgVersion = pkg && pkg.version ? pkg.version : 'unknown';
    } catch (_e) { }

    res.json({
        name: 'jw-backend',
        version: pkgVersion,
        buildTime: new Date().toISOString(),
        // Railway 常见注入（不保证存在）
        railway: {
            environment: process.env.RAILWAY_ENVIRONMENT || '',
            service: process.env.RAILWAY_SERVICE_NAME || '',
            gitCommit: process.env.RAILWAY_GIT_COMMIT_SHA || process.env.RAILWAY_GIT_COMMIT || ''
        },
        features: {
            timetableFollowRedirects: true,
            syncAwaitTimetable: true
        }
    });
});

/**
 * 同步接口：登录并同步所有数据到数据库
 * POST /api/sync
 * Body: { username, password }
 */
app.post('/api/sync', async (req, res) => {
    const { username, password, semester } = req.body;

    if (!username || !password) {
        return res.status(400).json({ success: false, message: '请提供学号和密码' });
    }

    // 检查数据库连接状态
    if (!dbReady) {
        return res.status(503).json({ success: false, message: '数据库未就绪，请稍后重试' });
    }

    try {
        console.log(`正在同步学生数据: ${username}${semester ? ` (学期: ${semester})` : ''}`);
        const loginResult = await login(username, password);

        if (!loginResult.success) {
            return res.status(401).json({ success: false, message: loginResult.message });
        }

        const cookies = loginResult.cookies;

        // 异步执行同步逻辑，不阻塞响应（或者等待完成，取决于需求）
        // 这里我们选择等待基本信息完成，其他后台同步
        const info = await getStudentInfo(cookies);
        if (info) {
            await syncStudent(username, info);

            // 关键：课表是前端“立即可见”的核心数据。
            // 之前是后台同步，前端立刻 GET /api/student/:id 经常拿到 courses=[]
            // 这里至少等待课表抓取+入库完成后再返回。
            let timetableCount = 0;
            let timetableDebug = { ok: false, reason: 'not_started' };
            let currentSemester = semester; // 默认为传入的学期
            try {
                // 如果指定了学期，带入学期参数抓取
                const timetable = await getTimetable(cookies, semester);
                if (timetable && Array.isArray(timetable) && timetable.length > 0) {
                    await syncCourses(username, timetable);
                    timetableCount = timetable.length;
                    timetableDebug = { ok: true, reason: 'synced' };
                    
                    // 从课表中获取当前学期
                    if (timetable[0] && timetable[0].semester) {
                        currentSemester = timetable[0].semester;
                        console.log(`从课表中获取到当前学期: ${currentSemester}`);
                    }
                } else if (Array.isArray(timetable) && timetable.length === 0) {
                    timetableDebug = { ok: false, reason: 'parsed_empty' };
                } else {
                    timetableDebug = { ok: false, reason: 'fetch_failed_or_null' };
                }
            } catch (e) {
                console.error('Sync courses failed (awaited):', e);
                timetableDebug = { ok: false, reason: 'exception', message: String(e && e.message ? e.message : e) };
            }
            
            // 后台静默同步其他数据
            // 注意：这里不使用 await，让它在后台运行
            // 如果需要确保所有数据都同步完再返回，可以在这里加 await
            // 但考虑到爬虫速度，建议先返回基本信息，其他让前端轮询或者下次进入时获取
            
            // 为了防止 Promise.all 抛出未捕获异常导致进程崩溃，这里单独处理每个 Promise
            const syncTasks = [
                getGrades(cookies).then(data => {
                    if (data) {
                        syncGrades(username, data);
                        // 统计所有学期的成绩数量
                        let totalGrades = 0;
                        for (const semester in data) {
                            if (data[semester] && Array.isArray(data[semester])) {
                                totalGrades += data[semester].length;
                            }
                        }
                        return totalGrades;
                    }
                    return 0;
                }).catch(e => {
                    console.error('Sync grades failed:', e);
                    return 0;
                }),
                getExamSchedule(cookies, currentSemester).then(data => {
                    if (data) {
                        syncExams(username, data);
                        return data.length;
                    }
                    return 0;
                }).catch(e => {
                    console.error('Sync exams failed:', e);
                    return 0;
                }),
                getSemesterPlan(cookies).then(data => {
                    if (data) {
                        syncPlans(username, data);
                        // 统计所有学期的培养计划数量
                        let totalPlans = 0;
                        for (const semester in data) {
                            if (data[semester] && Array.isArray(data[semester])) {
                                totalPlans += data[semester].length;
                            }
                        }
                        return totalPlans;
                    }
                    return 0;
                }).catch(e => {
                    console.error('Sync plans failed:', e);
                    return 0;
                }),
                getStudyProgress(cookies).then(data => {
                    if (data) {
                        syncProgress(username, data);
                        return 1; // 学分进度通常是一个对象，不是数组
                    }
                    return 0;
                }).catch(e => {
                    console.error('Sync progress failed:', e);
                    return 0;
                })
            ];
            
            // 触发任务但不等待
            Promise.all(syncTasks).then((results) => {
                // 统计各项数据
                const gradesCount = results[0] || 0;
                const examsCount = results[1] || 0;
                const plansCount = results[2] || 0;
                const progressCount = results[3] || 0;
                
                console.log(`后台同步完成: ${username}`);
                console.log(`数据统计: 课表=${timetableCount}, 成绩=${gradesCount}, 考试安排=${examsCount}, 培养计划=${plansCount}, 学分进度=${progressCount}`);
            });

            return res.json({
                success: true,
                message: timetableCount > 0
                    ? '登录成功，课表已同步，其它数据正在后台同步中'
                    : '登录成功，课表暂未同步到数据（请稍后在“同步”重试），其它数据正在后台同步中',
                student: info,
                timetableCount,
                timetableDebug
            });
        } else {
            return res.status(500).json({ success: false, message: '获取学生信息失败' });
        }
    } catch (error) {
        console.error('Sync error:', error);
        // 确保返回具体的错误信息以便调试
        const errorMessage = error && error.message ? error.message : String(error);
        const errorStack = error && error.stack ? error.stack : '';
        console.error('Error details:', { message: errorMessage, stack: errorStack });
        res.status(500).json({ 
            success: false, 
            message: '服务器内部错误: ' + errorMessage 
        });
    }
});

/**
 * 查询接口：从数据库获取已缓存的学生信息
 * GET /api/student/:id
 */
app.get('/api/student/:id', async (req, res) => {
    const studentId = req.params.id;
    const { semester } = req.query; // 获取查询参数中的学期

    try {
        const student = await Student.findByPk(studentId, {
            include: [
                // 如果设置了关联可以 include，目前模型是独立的，我们手动查询
            ]
        });

        if (!student) {
            return res.status(404).json({ success: false, message: '未找到该学生缓存数据' });
        }

        // 构造查询条件
        const where = { studentId };
        if (semester) {
            where.semester = semester;
        }

        // 查询关联数据
        const [courses, grades, exams, plans, progress] = await Promise.all([
            // 课表：按学期 -> 周次 -> 星期 -> 节次排序
            Course.findAll({
                where, // 如果指定了学期，则只返回该学期的课表
                order: [
                    ['semester', 'ASC'],
                    ['week', 'ASC'],
                    ['dayOfWeek', 'ASC'],
                    ['period', 'ASC'],
                    ['name', 'ASC']
                ]
            }),
            // 成绩：按学期 -> 课程编号排序
            Grade.findAll({
                where: { studentId }, // 成绩通常需要看全部，不强制过滤
                order: [
                    ['semester', 'ASC'],
                    ['courseCode', 'ASC']
                ]
            }),
            // 考试：按时间 -> 课程名排序
            Exam.findAll({
                where: { studentId },
                order: [
                    ['examTime', 'ASC'],
                    ['courseName', 'ASC']
                ]
            }),
            // 学期计划：按学期 -> 课程编号排序
            Plan.findAll({
                where: { studentId },
                order: [
                    ['semester', 'ASC'],
                    ['courseCode', 'ASC']
                ]
            }),
            // 学习完成情况：按分类名排序
            Progress.findAll({
                where: { studentId },
                order: [['category', 'ASC']]
            })
        ]);

        res.json({
            success: true,
            data: {
                info: student,
                courses,
                grades,
                exams,
                plans,
                progress
            }
        });
    } catch (error) {
        res.status(500).json({ success: false, message: error.message });
    }
});

/**
 * 学生列表（按学号升序）
 * GET /api/students
 */
app.get('/api/students', async (_req, res) => {
    try {
        const students = await Student.findAll({
            order: [['studentId', 'ASC']]
        });
        res.json({ success: true, data: students });
    } catch (error) {
        res.status(500).json({ success: false, message: error.message });
    }
});

app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server is running on port ${PORT}`);
});
