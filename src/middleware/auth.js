const jwt = require('jsonwebtoken');

const JWT_SECRET = process.env.JWT_SECRET || 'qinxu-jwt-secret-key-2026';
const JWT_EXPIRES_IN = process.env.JWT_EXPIRES_IN || '7d';

function generateToken(payload) {
    const tokenPayload = {
        userId: payload.username || payload.studentId,
        username: payload.username,
        studentId: payload.studentId || payload.username,
        iat: Math.floor(Date.now() / 1000),
        exp: Math.floor(Date.now() / 1000) + (7 * 24 * 60 * 60)
    };

    return jwt.sign(tokenPayload, JWT_SECRET, { algorithm: 'HS256' });
}

function verifyToken(token) {
    try {
        if (!token) {
            return { valid: false, error: '未提供认证令牌' };
        }

        const decoded = jwt.verify(token, JWT_SECRET);
        return { valid: true, data: decoded };
    } catch (error) {
        if (error.name === 'TokenExpiredError') {
            return { valid: false, error: '认证令牌已过期，请重新登录', code: 'TOKEN_EXPIRED' };
        }
        if (error.name === 'JsonWebTokenError') {
            return { valid: false, error: '无效的认证令牌', code: 'INVALID_TOKEN' };
        }
        return { valid: false, error: '认证失败: ' + error.message };
    }
}

function authenticate(req, res, next) {
    const authHeader = req.headers.authorization;

    if (!authHeader || !authHeader.startsWith('Bearer ')) {
        return res.status(401).json({
            success: false,
            message: '未提供有效的认证令牌',
            code: 'NO_TOKEN'
        });
    }

    const token = authHeader.substring(7);
    const result = verifyToken(token);

    if (!result.valid) {
        return res.status(401).json({
            success: false,
            message: result.error,
            code: result.code || 'AUTH_FAILED'
        });
    }

    req.user = result.data;
    next();
}

function optionalAuth(req, res, next) {
    const authHeader = req.headers.authorization;

    if (authHeader && authHeader.startsWith('Bearer ')) {
        const token = authHeader.substring(7);
        const result = verifyToken(token);

        if (result.valid) {
            req.user = result.data;
        }
    }

    next();
}

module.exports = {
    generateToken,
    verifyToken,
    authenticate,
    optionalAuth,
    JWT_SECRET,
    JWT_EXPIRES_IN
};
