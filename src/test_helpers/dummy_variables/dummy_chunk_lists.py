"""dummy_chunk_lists.py
Dummy lists of DocumentChunks to use for testing.
"""

from src.models import DocumentChunk
from src.test_helpers.mock_resume_generator import (
    MockResumeGenerator,
    ChunkValues,
    ChunkTemplates,
    DUMMY_RESUME_BLOCKS,
    DEFAULT_CHUNK_ORDER,
    DEFAULT_TEST_CHUNK_SIZE
)

# ---------------------------------------------------------------------------
# Setup dummy examples for testing
# ---------------------------------------------------------------------------

MOCK_DOCUMENT_CHUNK_LIST = [DocumentChunk(0, "text")]

# Basic Resume with default values (0 for all options in DUMMY_RESUME_BLOCKS)
MOCK_RESUME_GENERATOR_0 = MockResumeGenerator(
    chunk_values=ChunkValues(
        name="John Doe",
        email="john.doe@example.com",
        phone="123-456-7890",
        linkedin_name="john_doe23",
        skills=DUMMY_RESUME_BLOCKS["skills_filled"][0],
        company_name="Comcast",
    ),
    chunk_templates=ChunkTemplates(
        contact_info=DUMMY_RESUME_BLOCKS["contact_info"][0],
        work_experience=DUMMY_RESUME_BLOCKS["work_experience"][0],
        education=DUMMY_RESUME_BLOCKS["education"][0],
        projects=DUMMY_RESUME_BLOCKS["projects"][0],
        skills=DUMMY_RESUME_BLOCKS["skills_fillable"][0],
        other=None,  # optional
    ),
    chunk_order=DEFAULT_CHUNK_ORDER,  # e.g., ["contact_info", "work_experience", "education", "projects", "skills"]
    chunk_size=DEFAULT_TEST_CHUNK_SIZE,
)

# Basic Resume with 1 for all DUMMY_RESUME_BLOCKS options and a scrambled order
MOCK_RESUME_GENERATOR_1 = MockResumeGenerator(
    chunk_values=ChunkValues(
        name="Carlos Mendez",
        email="c.mendez@company.net",
    ),
    chunk_templates=ChunkTemplates(
        contact_info = DUMMY_RESUME_BLOCKS["contact_info"][1],
        work_experience = DUMMY_RESUME_BLOCKS["work_experience"][1],
        education = DUMMY_RESUME_BLOCKS["education"][1],
        projects = DUMMY_RESUME_BLOCKS["projects"][1],
        skills = DUMMY_RESUME_BLOCKS["skills_filled"][1]
    ),
    chunk_order = [
        "work_experience",
        "education",
        "contact_info", # Scanner could read order wrong if format is strange
        "skills",
        "projects",
    ]
)

# Basic Resume with 2 for all DUMMY_RESUME_BLOCKS options and a scrambled order
MOCK_RESUME_GENERATOR_2 = MockResumeGenerator(
    chunk_values=ChunkValues(
        name="Alice Lee",
        email="alice.lee@example.co.uk",
    ),
    chunk_templates=ChunkTemplates(
        contact_info = DUMMY_RESUME_BLOCKS["contact_info"][2],
        work_experience = DUMMY_RESUME_BLOCKS["work_experience"][2],
        education = DUMMY_RESUME_BLOCKS["education"][2],
        projects = DUMMY_RESUME_BLOCKS["projects"][2],
        skills = DUMMY_RESUME_BLOCKS["skills_filled"][2]
    ),
    chunk_order = [
        "contact_info",
        "skills",
        "education",
        "work_experience"
        "projects",
    ]
)


# Setup Test Persons to test with
MOCK_PERSONS = [
    {"name": "John Doe", "email": "john.doe@example.com", "linkedin_name": "john_doe"},
    {"name": "John E. Doe", "email": "j.e.doe@protonmail.com", "linkedin_name": "john-e-doe"},
    {"name": "John Edward Doe", "email": "jedward.doe@outlook.co.uk", "linkedin_name": "johnedwarddoe"},
    {"name": "John Doe-Smith", "email": "john.doe-smith@smithfamily.org", "linkedin_name": "john_doe_smith"},
    {"name": "María-José Carreño", "email": "mariajose.carreno@gmail.es", "linkedin_name": "maria-jose-carreno"},
    {"name": "Nguyễn Văn An", "email": "nguyenvan.an@vnmail.vn", "linkedin_name": "nguyen-van-an"},
    {"name": "Zhang Wei", "email": "zhang.wei@aliyun.cn", "linkedin_name": "zhangwei88"},
    {"name": "Mei-Ling Yi", "email": "mei.ling.yi@ntu.edu.sg", "linkedin_name": "mei-ling-yi"},
    {"name": "Oluwaseun Adeyemi", "email": "oluwaseun.a@lagosconnect.com", "linkedin_name": "oluwaseun-adeyemi"},
    {"name": "Ivan Ivanovich Petrov", "email": "ivan.petrov@ya.ru", "linkedin_name": "ivan-ivanovich-petrov"},
    {"name": "Anne-Marie O'Neill", "email": "annemarie.oneill@irishmail.ie", "linkedin_name": "anne-marie-oneill"},
    {"name": "Chloé Dubois", "email": "chloe.dubois@orange.fr", "linkedin_name": "chloe_dubois"},
    {"name": "Søren Kierkegaard", "email": "soren.kierkegaard@cph.dk", "linkedin_name": "soren-kierkegaard"},
    {"name": "Ji-hoon Park", "email": "jihoon.park@korea.kr", "linkedin_name": "jihoonpark_official"},
    {"name": "Yuki Takahashi", "email": "yuki.takahashi@me.jp", "linkedin_name": "yuki-takahashi"},
    {"name": "Priya Kaur Singh", "email": "priya.ks@outlook.in", "linkedin_name": "priya-kaur-singh"},
    {"name": "Arjun Srinivasan", "email": "arjun.srinivasan@iitm.ac.in", "linkedin_name": "arjun-srinivasan"},
    {"name": "Fatima Al-Sayed", "email": "fatima.al.sayed@dubai.ae", "linkedin_name": "fatimaal-sayed"},
    {"name": "Javier Fernandez Garcia", "email": "javier.fg@correo.es", "linkedin_name": "javier-fernandez-garcia"},
    {"name": "Daan van der Beek", "email": "daan.vanderbeek@kpn.nl", "linkedin_name": "daan-van-der-beek"},
    {"name": "Dr. Felicity Shaw", "email": "dr.felicity.shaw@nhs.uk", "linkedin_name": "dr-felicity-shaw"},
    {"name": "William (Will) Smith", "email": "will.smith@hollywoodmail.com", "linkedin_name": "will-smith-real"},
    {"name": "Michael Johnson Jr.", "email": "michael.johnsonjr@utexas.edu", "linkedin_name": "michael-johnson-jr"},
    {"name": "M. K. Gandhi", "email": "m.k.gandhi@freedom.org", "linkedin_name": "mk-gandhi"},
]
