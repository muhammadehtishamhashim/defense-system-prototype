"""
Sample threat intelligence data for testing and demonstration.
"""

# Sample JSON threat feed
SAMPLE_JSON_FEED = {
    "indicators": [
        {
            "type": "ip",
            "value": "192.168.1.100",
            "context": "Malware command and control server detected"
        },
        {
            "type": "domain",
            "value": "malicious-domain.com",
            "context": "Phishing site hosting malware"
        },
        {
            "type": "hash",
            "value": "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
            "context": "SHA-256 hash of ransomware executable"
        }
    ],
    "timestamp": "2024-01-15T10:30:00Z",
    "source": "Sample Threat Feed"
}

# Sample CSV threat feed
SAMPLE_CSV_FEED = """indicator_type,indicator_value,description,confidence
ip,10.0.0.1,Suspicious IP address scanning network,0.8
domain,phishingsite.net,Fake banking site,0.9
cve,CVE-2023-1234,High severity vulnerability in web server,0.95"""

# Sample XML threat feed
SAMPLE_XML_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<threat_feed>
    <indicator>
        <type>ip</type>
        <value>203.0.113.1</value>
        <description>Known botnet command server</description>
        <confidence>0.95</confidence>
    </indicator>
    <indicator>
        <type>hash</type>
        <value>d4735e3a265e16eee03f59718b9b5d03019c07d8b6c51f90da3a666eec13ab35</value>
        <description>MD5 hash of malicious file</description>
        <confidence>0.85</confidence>
    </indicator>
</threat_feed>"""

# Sample text threat feed
SAMPLE_TEXT_FEED = """
Threat Intelligence Report - January 2024

Recent malware campaign detected:
- C2 Server IP: 45.67.89.123 (high confidence)
- Phishing Domain: fakebank-login.com (medium confidence)
- Malicious Hash: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 (SHA-256)

CVE-2024-0123 affects multiple systems and should be patched immediately.
Email address used in campaign: attacker@example.com
"""

# Sample IOC objects for testing
SAMPLE_IOCS = [
    {
        "type": "ip",
        "value": "192.168.1.100",
        "context": "Command and control server for malware campaign"
    },
    {
        "type": "domain",
        "value": "malware-site.net",
        "context": "Distribution site for ransomware"
    },
    {
        "type": "hash",
        "value": "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
        "context": "SHA-256 of malicious executable"
    },
    {
        "type": "cve",
        "value": "CVE-2023-4444",
        "context": "Critical vulnerability in SSL library"
    }
]
