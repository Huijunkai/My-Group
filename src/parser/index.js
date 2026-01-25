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
                    const timeLineIndex = lines.findIndex(l => l.includes('[') && l.includes('节]'));
                    
                    // 解析周次信息
                    let startWeek = 0, endWeek = 0;
                    let isOdd = false, isEven = false;
                    
                    // 提取周次字符串，如 "1-16" 或 "1-16(单)"
                    let weekStr = lines[timeLineIndex] || '';
                    
                    // 优化匹配逻辑：优先匹配带"周"字的，如果没有则尝试匹配纯数字范围（通常在方括号内）
                    // 强智系统常见格式：
                    // 1. [1-16周]
                    // 2. [1-16周](单)
                    // 3. 1-16周
                    
                    // 提取方括号内的内容作为周次依据
                    const bracketMatch = weekStr.match(/\[(.*?)\]/);
                    let weekContent = bracketMatch ? bracketMatch[1] : weekStr;
                    
                    // 如果内容包含"节"，说明可能提取错了或者是混合信息，需要进一步清洗
                    // 这里我们假设周次信息通常包含"周"字，或者纯数字范围
                    
                    // 判断单双周
                    if (weekContent.includes('单')) isOdd = true;
                    if (weekContent.includes('双')) isEven = true;
                    
                    // 提取周次范围 (匹配 "数字-数字" 且后面紧跟 "周" 或者 位于方括号内)
                    // 优先匹配带 "周" 的
                    let rangeMatch = weekContent.match(/(\d+)-(\d+)周/);
                    if (!rangeMatch) {
                        // 如果没有"周"字，尝试匹配纯数字范围，但要排除可能是节次的情况
                        // 通常节次会带有"节"字，或者在周次之后
                        rangeMatch = weekContent.match(/(\d+)-(\d+)/);
                    }

                    if (rangeMatch) {
                        startWeek = parseInt(rangeMatch[1]);
                        endWeek = parseInt(rangeMatch[2]);
                    } else {
                        // 可能是单个周，如 [5周]
                        const singleMatch = weekContent.match(/(\d+)周/);
                        if (singleMatch) {
                            startWeek = endWeek = parseInt(singleMatch[1]);
                        }
                    }

                    // 解析节次信息
                    let periodStr = '';
                    let startPeriod = 0, endPeriod = 0;
                    
                    // 节次通常带有 "节" 字，或者在周次信息之后
                    // 匹配 "数字-数字节"
                    const periodMatch = weekStr.match(/(\d+)-(\d+)节/);
                    if (periodMatch) {
                        periodStr = periodMatch[0];
                        startPeriod = parseInt(periodMatch[1]);
                        endPeriod = parseInt(periodMatch[2]);
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
                                 startPeriod = s;
                                 endPeriod = e;
                                 periodStr = `${s}-${e}节`;
                             }
                        }
                    }

                    courses.push({
                        semester: semester,
                        dayOfWeek: dayOfWeek,
                        name: lines[0], // 第一行通常是课程名
                        teacher: lines[1], // 第二行通常是老师
                        weeks: weekStr, // 原始周次信息
                        startWeek,
                        endWeek,
                        isOdd,
                        isEven,
                        period: periodStr,
                        startPeriod,
                        endPeriod,
                        // 地点通常在“节”那一行的下一行
                        location: timeLineIndex !== -1 && lines[timeLineIndex + 1] ? lines[timeLineIndex + 1] : (lines[3] || '未知'),
                        raw: lines.join(' | ')
                    });
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
