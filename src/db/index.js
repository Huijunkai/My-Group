const { Sequelize } = require('sequelize');

const sequelize = new Sequelize('postgresql://postgres:eKtmtaNElgnVqHPBTQNIyyAjvLdcUBFR@switchyard.proxy.rlwy.net:51088/railway', {
    dialect: 'postgres',
    logging: false, // 设置为 console.log 可以查看 SQL 语句
    pool: {
        max: 5,
        min: 0,
        acquire: 30000,
        idle: 10000
    }
});

// 测试连接并同步模型
async function initDatabase() {
    try {
        await sequelize.authenticate();
        console.log('数据库连接成功');
        // sync({ alter: true }) 会根据模型定义自动更新表结构
        await sequelize.sync({ alter: true });
        console.log('所有模型已同步');

        // 仅对 Course(课程表) 做结构收敛：移除不再需要的拆分字段列
        // 说明：sequelize.sync({ alter: true }) 在部分情况下不会自动 drop 列，这里显式处理一次
        try {
            const { QueryTypes } = require('sequelize');
            const qi = sequelize.getQueryInterface();
            const columns = await qi.describeTable('Courses');
            const deprecatedColumns = ['startWeek', 'endWeek', 'isOdd', 'isEven', 'startPeriod', 'endPeriod'];
            for (const col of deprecatedColumns) {
                if (columns[col]) {
                    await qi.removeColumn('Courses', col);
                }
            }
            console.log('Course 表结构已收敛');

            // 数据清洗：确保 weeks 不再包含“节次”，节次进入 period 列
            // 例如：weeks="6(全部)[01-02节]" -> weeks="6(全部)", period="1-2节"
            const dirtyRows = await sequelize.query(
                'SELECT "studentId", "semester", "name", "dayOfWeek", "weeks", "period" FROM "Courses" WHERE "weeks" LIKE :pat',
                { replacements: { pat: '%节%' }, type: QueryTypes.SELECT }
            );

            if (dirtyRows.length > 0) {
                let cleaned = 0;
                let skipped = 0;
                for (const row of dirtyRows) {
                    const weeks = String(row.weeks || '');
                    const m = weeks.match(/(\d{1,2})-(\d{1,2})节/);
                    if (!m) {
                        skipped++;
                        continue;
                    }

                    const cleanedWeeks = weeks.replace(/\[?\d{1,2}-\d{1,2}节\]?/g, '').trim();
                    const period = (row.period && String(row.period).trim())
                        ? String(row.period).trim()
                        : `${m[1]}-${m[2]}节`; // 保留前导 0

                    await sequelize.query(
                        `UPDATE "Courses"
                         SET "weeks" = :weeks, "period" = :period
                         WHERE "studentId" = :studentId
                           AND "semester" = :semester
                           AND "name" = :name
                           AND "dayOfWeek" = :dayOfWeek`,
                        {
                            replacements: {
                                weeks: cleanedWeeks,
                                period,
                                studentId: row.studentId,
                                semester: row.semester,
                                name: row.name,
                                dayOfWeek: row.dayOfWeek
                            }
                        }
                    );
                    cleaned++;
                }
                console.log(`Course 表 weeks 清洗完成：dirty=${dirtyRows.length}, cleaned=${cleaned}, skipped=${skipped}`);
            }

            // 进一步规范化：按你的规则保留“全部”(不保留括号)，移除 (单)/(双) 等；并统一分隔符
            const normalizeWeeks = (input) => {
                if (!input) return '';
                let s = String(input).trim();
                s = s.replace(/\s+/g, '');
                s = s.replace(/（/g, '(').replace(/）/g, ')');
                s = s.replace(/[，、]/g, ',');
                s = s.replace(/[～—–－]/g, '-');
                s = s.replace(/至/g, '-');
                s = s.replace(/周/g, '');
                // 保留“全部”(不保留括号)，删除其他括号内容
                let hasAll = false;
                s = s.replace(/\((.*?)\)/g, (_m, inner) => {
                    if (String(inner).includes('全部')) {
                        hasAll = true;
                        return '全部';
                    }
                    return '';
                });
                s = s.replace(/第/g, '').replace(/单/g, '').replace(/双/g, '');
                // 提取数字范围串
                const m = s.match(/[0-9]{1,2}(?:-[0-9]{1,2})?(?:,[0-9]{1,2}(?:-[0-9]{1,2})?)*/);
                if (m && m[0]) s = m[0] + (hasAll ? '全部' : '');
                else if (hasAll) s = '全部';
                else s = s.replace(/[^0-9,-]/g, '');
                // 压缩连续分隔符
                s = s.replace(/,+/g, ',').replace(/-+/g, '-');
                s = s.replace(/^,|,$/g, '');
                s = s.replace(/^-|-$/g, '');
                return s;
            };

            const normCandidates = await sequelize.query(
                'SELECT "studentId", "semester", "name", "dayOfWeek", "weeks" FROM "Courses" WHERE "weeks" IS NOT NULL AND "weeks" <> :empty',
                { replacements: { empty: '' }, type: QueryTypes.SELECT }
            );

            let normalized = 0;
            for (const row of normCandidates) {
                const before = String(row.weeks || '');
                const after = normalizeWeeks(before);
                if (after && after !== before) {
                    await sequelize.query(
                        `UPDATE "Courses"
                         SET "weeks" = :weeks
                         WHERE "studentId" = :studentId
                           AND "semester" = :semester
                           AND "name" = :name
                           AND "dayOfWeek" = :dayOfWeek`,
                        {
                            replacements: {
                                weeks: after,
                                studentId: row.studentId,
                                semester: row.semester,
                                name: row.name,
                                dayOfWeek: row.dayOfWeek
                            }
                        }
                    );
                    normalized++;
                }
            }
            if (normalized > 0) {
                console.log(`Course 表 weeks 规范化完成：updated=${normalized}`);
            }

            // 用 raw 回填（修复曾经把 (全部) 等信息误删导致“周次获取不到/不完整”）
            const extractWeeksPeriodFromRaw = (raw) => {
                if (!raw) return null;
                const str = String(raw);
                // 找到包含 [xx-xx节] 的那一段（raw 是用 ' | ' 连接的）
                const segMatch = str.match(/\|\s*([^|]*\[\d{1,2}-\d{1,2}节\][^|]*)\s*\|/);
                const seg = segMatch ? segMatch[1].trim() : '';
                if (!seg) return null;
                const pMatch = seg.match(/\[(\d{1,2})-(\d{1,2})节\]/);
                if (!pMatch) return null;
                const period = `${pMatch[1]}-${pMatch[2]}节`;
                const weekPart = seg.replace(pMatch[0], '').trim();
                const weeks = normalizeWeeks(weekPart);
                return { weeks, period };
            };

            const rawRows = await sequelize.query(
                'SELECT "studentId", "semester", "name", "dayOfWeek", "weeks", "period", "raw" FROM "Courses" WHERE "raw" IS NOT NULL AND "raw" <> :empty',
                { replacements: { empty: '' }, type: QueryTypes.SELECT }
            );

            let backfilled = 0;
            for (const row of rawRows) {
                const extracted = extractWeeksPeriodFromRaw(row.raw);
                if (!extracted) continue;
                const weeksBefore = String(row.weeks || '');
                const periodBefore = String(row.period || '');
                // 只有在提取到有效值且不同的时候才更新
                if ((extracted.weeks && extracted.weeks !== weeksBefore) || (extracted.period && extracted.period !== periodBefore)) {
                    await sequelize.query(
                        `UPDATE "Courses"
                         SET "weeks" = :weeks, "period" = :period
                         WHERE "studentId" = :studentId
                           AND "semester" = :semester
                           AND "name" = :name
                           AND "dayOfWeek" = :dayOfWeek`,
                        {
                            replacements: {
                                weeks: extracted.weeks || weeksBefore,
                                period: extracted.period || periodBefore,
                                studentId: row.studentId,
                                semester: row.semester,
                                name: row.name,
                                dayOfWeek: row.dayOfWeek
                            }
                        }
                    );
                    backfilled++;
                }
            }
            if (backfilled > 0) {
                console.log(`Course 表 raw 回填完成：updated=${backfilled}`);
            }
        } catch (e) {
            console.warn('Course 表结构收敛失败（可忽略）:', e.message);
        }
    } catch (error) {
        console.error('数据库连接或同步失败:', error);
    }
}

module.exports = {
    sequelize,
    initDatabase
};
