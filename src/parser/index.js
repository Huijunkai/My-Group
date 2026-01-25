const cheerio = require('cheerio');

/**
 * 解析学生个人信息
 */
function parseStudentInfo(html) {
    const $ = cheerio.load(html);
    const studentInfo = {
        name: '',
        gender: '',
        enrollmentYear: '',
        className: '',
        major: '',
        college: ''
    };

    const $table = $('#xjkpTable');
    if ($table.length === 0) return null;

    studentInfo.name = $table.find('td:contains("姓名")').first().next('td').text().replace(/\s+/g, '');
    studentInfo.gender = $table.find('td:contains("性别")').first().next('td').text().replace(/\s+/g, '');

    $table.find('td').each((i, el) => {
        const text = $(el).text().trim();
        if (text.includes('院系：')) {
            studentInfo.college = text.split('：')[1].trim();
        } else if (text.includes('专业：')) {
            studentInfo.major = text.split('：')[1].trim();
        } else if (text.includes('班级：')) {
            studentInfo.className = text.split('：')[1].trim();
        }
    });

    studentInfo.enrollmentYear = $table.find('td:contains("入学日期")').next('td').text().replace(/\s+/g, '');

    return studentInfo;
}

/**
 * 解析课表信息
 */
function parseTimetable(html) {
    const $ = cheerio.load(html);
    const courses = [];

    let semester = $('#xnxqh option[selected]').text().trim() || 
                   $('.Nsb_right_title_sj').text().trim() || 
                   $('option[selected]').first().text().trim() || 
                   '未知学期';
    
    const $table = $('#kbtable');
    
    $table.find('td div.kbcontent').each((i, el) => {
        const content = $(el).html();
        if (content && content.trim() && content !== '&nbsp;') {
            const parts = content.split('---------------------');
            
            // 通过单元格在行中的索引来确定星期几
            // 索引 1 是周一，2 是周二...
            const columnIndex = $(el).closest('td').index();
            const weekDays = ['', '星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日'];
            const dayOfWeek = weekDays[columnIndex] || '未知';

            parts.forEach(part => {
                const lines = part.split('<br>').map(line => cheerio.load(line).text().trim()).filter(line => line);
                if (lines.length >= 3) {
                    // 强智系统特征：包含 [xx-xx节] 的那一行一定是时间/周次信息
                    let timeLineIndex = lines.findIndex(l => l.includes('[') && l.includes('节]'));
                    if (timeLineIndex === -1) {
                        // 兜底：有些页面不含 "节]"，但仍会在同一行包含周次/节次信息
                        timeLineIndex = lines.findIndex(l => (l.includes('周') || l.includes('节') || (l.includes('[') && l.includes(']'))));
                    }
                    
                    // 提取时间/周次那一行原文
                    const weekStr = timeLineIndex !== -1 ? (lines[timeLineIndex] || '') : '';
                    const timeLine = (weekStr || '').replace(/\s+/g, '');
                    
                    // 解析节次信息
                    let periodStr = '';
                    // 优先：节次用中括号括起来
                    const periodBracketMatch = timeLine.match(/\[(\d{1,2})-(\d{1,2})节\]/);
                    if (periodBracketMatch) {
                        // 保留前导 0（如 01-02）
                        periodStr = `${periodBracketMatch[1]}-${periodBracketMatch[2]}节`;
                    }
                    
                    // 节次通常带有 "节" 字，或者在周次信息之后
                    // 匹配 "数字-数字节"
                    const periodMatch = weekStr.match(/(\d+)-(\d+)节/);
                    if (periodMatch) {
                        periodStr = periodMatch[0];
                    } else {
                        // 如果没有 "节" 字，尝试查找周次之后的数字对
                        // 例如：[1-16周]01-02
                        // 先去掉周次部分
                        const contentAfterWeek = weekStr.replace(/\[.*?\]|.*?周/g, '');
                        const numMatch = contentAfterWeek.match(/(\d+)-(\d+)/);
                        if (numMatch) {
                             // 简单的启发式：节次通常小于 14
                             const s = parseInt(numMatch[1]);
                             const e = parseInt(numMatch[2]);
                             if (s <= 14 && e <= 14) {
                                 // 原始可能带前导 0，这里直接使用匹配文本
                                 periodStr = `${numMatch[1]}-${numMatch[2]}节`;
                             }
                        }
                    }

                    // 解析周次信息（保证 weeks 不含节次）
                    // 常见格式：
                    // - [1-16周](单) + [01-02节]
                    // - 6(全部)[01-02节]   （周次不带“周”字）
                    // - [1-8,10-16周](双)01-02
                    let weeksOnlySource = timeLine;
                    if (periodBracketMatch) {
                        weeksOnlySource = weeksOnlySource.replace(periodBracketMatch[0], '');
                    }
                    // 去掉纯数字节次（无“节”字）的情况：01-02 / 1-2
                    if (periodStr) {
                        const p = periodStr.replace('节', '');
                        weeksOnlySource = weeksOnlySource.replace(p, '');
                    }

                    // 解析周次：只保留简单的分隔（数字/逗号/短横线），不要单双周等标记
                    // 输出示例：
                    // - "1-16"
                    // - "1-8,10-16"
                    // - "6"
                    const normalizeWeeks = (input) => {
                        if (!input) return '';
                        let s = String(input).replace(/\s+/g, '');
                        // 中文括号统一
                        s = s.replace(/（/g, '(').replace(/）/g, ')');
                        // 统一分隔符：中文逗号/顿号 -> 逗号；各种连接符 -> 短横线
                        s = s.replace(/[，、]/g, ',');
                        s = s.replace(/[～—–－]/g, '-');
                        s = s.replace(/至/g, '-');
                        // 去掉节次残留
                        s = s.replace(/\[?\d{1,2}-\d{1,2}节\]?/g, '');
                        // 去掉“周”字
                        s = s.replace(/周/g, '');
                        // 去掉括号内描述：保留“全部”（不保留括号），移除 (单)/(双) 等
                        let hasAll = false;
                        s = s.replace(/\((.*?)\)/g, (_m, inner) => {
                            if (String(inner).includes('全部')) {
                                hasAll = true;
                                return '全部';
                            }
                            return '';
                        });
                        // 去掉其他文本标记（不需要单双周）
                        s = s.replace(/第/g, '').replace(/单/g, '').replace(/双/g, '');
                        // 提取数字范围串
                        const m = s.match(/[0-9]{1,2}(?:-[0-9]{1,2})?(?:,[0-9]{1,2}(?:-[0-9]{1,2})?)*/);
                        if (m && m[0]) return m[0] + (hasAll ? '全部' : '');
                        if (hasAll) return '全部';
                        // 兜底：保留数字/逗号/短横线 + (全部)
                        const fallback = s.replace(/[^0-9,\-全都部,]/g, '');
                        return fallback;
                    };

                    // 规则：节次在中括号里，剩余部分就是周次
                    // 例如：9-11,13(全部)[01-02节] -> weekExpr=9-11,13(全部)
                    const weekExpr = normalizeWeeks(weeksOnlySource);

                    // 将周次表达式拆分为单周数组（不考虑单双周等复杂规则：只按数字/逗号/短横线分隔）
                    const parseWeekList = (expr) => {
                        if (!expr) return [];
                        const s = String(expr).replace(/全部/g, '').replace(/[^0-9,\-]/g, '');
                        if (!s) return [];
                        const out = [];
                        const parts = s.split(',').filter(Boolean);
                        for (const part of parts) {
                            if (part.includes('-')) {
                                const [a, b] = part.split('-');
                                const start = parseInt(a, 10);
                                const end = parseInt(b, 10);
                                if (!Number.isNaN(start) && !Number.isNaN(end)) {
                                    const lo = Math.min(start, end);
                                    const hi = Math.max(start, end);
                                    for (let w = lo; w <= hi; w++) out.push(w);
                                }
                            } else {
                                const w = parseInt(part, 10);
                                if (!Number.isNaN(w)) out.push(w);
                            }
                        }
                        // 去重 + 排序
                        return Array.from(new Set(out)).sort((x, y) => x - y);
                    };

                    const weekList = parseWeekList(weekExpr);
                    const location = timeLineIndex !== -1 && lines[timeLineIndex + 1] ? lines[timeLineIndex + 1] : (lines[3] || '未知');
                    const base = {
                        semester: semester,
                        dayOfWeek: dayOfWeek,
                        name: lines[0], // 第一行通常是课程名
                        teacher: lines[1], // 第二行通常是老师
                        period: periodStr,  // 节次信息（区间，如 01-02节）
                        location,
                        raw: lines.join(' | ')
                    };

                    // 核心优化：按周拆分存储，每条记录对应一个 week
                    if (weekList.length > 0) {
                        weekList.forEach(weekNum => {
                            courses.push({
                                ...base,
                                week: weekNum,
                                weeks: String(weekNum) // weeks 字段保存单周，便于前端兼容
                            });
                        });
                    } else {
                        // 兜底：解析不到周次时仍入库一条（week=0 表示未知）
                        courses.push({
                            ...base,
                            week: 0,
                            weeks: '0'
                        });
                    }
                }
            });
        }
    });

    return courses;
}

/**
 * 解析成绩信息
 */
function parseGrades(html) {
    const $ = cheerio.load(html);
    const gradesGrouped = {};

    let $table = $('#dataList');
    if ($table.length === 0) {
        $table = $('table').filter((i, el) => $(el).text().includes('成绩') && $(el).find('tr').length > 1);
    }
    
    $table.find('tr').each((i, el) => {
        const tds = $(el).find('td');
        const firstTdText = $(tds[0]).text().trim();
        if (!firstTdText || firstTdText === '序号' || firstTdText.includes('课程')) return;

        if (tds.length >= 6) {
            const semester = $(tds[1]).text().trim();
            const gradeItem = {
                courseCode: $(tds[2]).text().trim(),
                courseName: $(tds[3]).text().trim(),
                score: $(tds[4]).text().trim(),
                credit: $(tds[5]).text().trim(),
                gradePoint: $(tds[6]).text().trim(),
                courseType: $(tds[7]).text().trim(),
                examType: $(tds[8]).text().trim()
            };

            if (!gradesGrouped[semester]) {
                gradesGrouped[semester] = [];
            }
            gradesGrouped[semester].push(gradeItem);
        }
    });

    return gradesGrouped;
}

/**
 * 解析考试安排
 */
function parseExams(html) {
    const $ = cheerio.load(html);
    const exams = [];
    const $table = $('#dataList');
    
    $table.find('tr').each((i, el) => {
        if (i === 0) return;
        const tds = $(el).find('td');
        if (tds.length >= 6) {
            exams.push({
                courseName: $(tds[1]).text().trim(),
                examTime: $(tds[3]).text().trim(),
                location: $(tds[4]).text().trim(),
                seatNumber: $(tds[5]).text().trim(),
                examType: $(tds[2]).text().trim(),
                status: $(tds[6]).text().trim()
            });
        }
    });

    return exams;
}

/**
 * 解析学期计划
 */
function parseSemesterPlan(html) {
    const $ = cheerio.load(html);
    const plansGrouped = {};
    const $table = $('#dataList');
    
    $table.find('tr').each((i, el) => {
        if (i === 0) return;
        const tds = $(el).find('td');
        if (tds.length >= 6) {
            const semester = $(tds[1]).text().trim();
            const planItem = {
                courseCode: $(tds[2]).text().trim(),
                courseName: $(tds[3]).text().trim(),
                credit: $(tds[4]).text().trim(),
                totalHours: $(tds[5]).text().trim(),
                courseType: $(tds[6]).text().trim(),
                examType: $(tds[7]).text().trim()
            };

            if (!plansGrouped[semester]) {
                plansGrouped[semester] = [];
            }
            plansGrouped[semester].push(planItem);
        }
    });

    return plansGrouped;
}

/**
 * 解析学习进度
 */
function parseStudyProgress(html) {
    const $ = cheerio.load(html);
    const progressData = [];

    const $table = $('table').filter((i, el) => {
        const headText = $(el).find('tr').first().text();
        return headText.includes('课程体系') && headText.includes('要求学分');
    }).first();
    
    $table.find('tr').each((i, el) => {
        const tds = $(el).find('td');
        if (tds.length < 5) return;
        const category = $(tds[0]).text().trim();
        if (!category || category === '课程体系(属性)') return;

        progressData.push({
            category: category,
            requiredCredits: $(tds[1]).text().trim(),
            completedCredits: $(tds[2]).text().trim(),
            currentCredits: $(tds[3]).text().trim(),
            remainingCredits: $(tds[4]).text().trim()
        });
    });

    return progressData;
}

module.exports = {
    parseStudentInfo,
    parseTimetable,
    parseGrades,
    parseExams,
    parseSemesterPlan,
    parseStudyProgress
};
