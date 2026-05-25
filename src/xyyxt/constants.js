module.exports = {
    BASE_URL: 'http://card.beitoucloud.com',
    AUTH_URL: 'http://card.beitoucloud.com/yxtapp/mobile/auth',
    ELEC_URL_NANNING: 'http://202.103.236.36:10002',
    ELEC_URL_GUILIN: 'http://221.7.150.20:10004',
    ROOM_URL_GUILIN: 'http://221.7.150.22:10005',
    ELEC_URL: 'http://202.103.236.36:10002',
    BASIC_AUTH: 'Basic bGV2aWFfY2xpZW50OmxldmlhX3NlY3JldA==',
    DEFAULT_HEADERS: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Origin': 'http://card.beitoucloud.com',
        'Referer': 'http://card.beitoucloud.com/yxth5/'
    },
    CAMPUS_CONFIG: {
        nnxq: {
            name: '南宁校区',
            elecUrl: 'http://202.103.236.36:10002',
            roomUrl: 'http://202.103.236.36:10002'
        },
        glxq: {
            name: '桂林校区',
            elecUrl: 'http://221.7.150.20:10004',
            roomUrl: 'http://221.7.150.22:10005'
        }
    },
    GUILIN_SERVER_1: 'http://221.7.150.22:10005',
    GUILIN_SERVER_2: 'http://221.7.150.20:10004',
    GUILIN_BUILDING_SERVER_MAP: {
        '4320': 'http://221.7.150.22:10005',
        '4509': 'http://221.7.150.22:10005',
        '4722': 'http://221.7.150.22:10005',
        '4812': 'http://221.7.150.22:10005',
        '6436': 'http://221.7.150.22:10005',
        '6819': 'http://221.7.150.22:10005',
        'B101': 'http://221.7.150.20:10004',
        'B102': 'http://221.7.150.20:10004',
        'B8': 'http://221.7.150.20:10004'
    }
};
