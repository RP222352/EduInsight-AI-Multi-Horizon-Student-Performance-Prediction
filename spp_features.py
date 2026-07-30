"""
spp_features.py — the knowledge base for every column in
"Academic Risk student performance Prediction.csv".

Nothing numeric lives here. Ranges, thresholds, pass rates and what counts as
"good" are all computed at runtime from the CSV (see spp_engine.py), because
this dataset is a merge of five different studies and each column is only
meaningful inside the study it came from.

Per feature:
  label       short human name
  question    what the wizard asks
  description what it actually measures, and how to read it
  unit        display unit
  actionable  can a school change it this term?
  prefer      +1 higher is better / -1 lower is better / None unknown
              (a sanity guard — if the data disagrees, the feature is left
               ungraded rather than producing harmful advice)
  levels      ordered {code: label} for coded ordinal fields
  order       display order for string categoricals (low -> high)
  immediate   short-term actions   (Recommendations)
  longterm    sustained changes    (Suggestions)
"""

# --------------------------------------------------------------------------- #
#  Columns that define the pass/fail outcome. If any of these reaches a model
#  it is reading the answer, not predicting it.
# --------------------------------------------------------------------------- #
KNOWN_LEAK_COLUMNS = {
    "grade_letter": "'D' is exactly the Fail class - the letter is derived from the same score as the target.",
    "G3": "The target is defined as G3 >= 10, so this column contains the answer.",
    "exam_score": "The target is defined as exam_score >= 60.",
    "avg_score": "The target is defined as avg_score >= 60.",
    "GPA": "The target is defined as GPA >= 2.0.",
    "result": "This is the target itself.",
    "fact_id": "Row identifier - carries no information about the student.",
    "source": "Which of the five merged studies the row came from - a data-provenance field.",
}

FEATURE_META = {
    # ===================================================== xAPI-Edu (LMS behaviour)
    "raisedhands": dict(
        label="Class participation (hands raised)",
        question="How many times has the student raised their hand to participate?",
        description="Number of times the student volunteered an answer or question in class "
                    "this term, logged by the teacher. It is a proxy for classroom engagement. "
                    "A quiet student is not automatically disengaged, but a count near zero "
                    "almost always means the student has stopped following lessons.",
        unit="times this term", actionable=True, prefer=1,
        immediate=["Ask the student one planned question per lesson, given 30 seconds in advance.",
                   "Use think-pair-share so speaking up carries no risk of being wrong alone."],
        longterm=["Build a classroom culture where wrong answers are treated as useful.",
                  "Credit written and small-group contributions for students who will not speak."],
    ),
    "visited_resources": dict(
        label="Course resources viewed",
        question="How many times has the student opened course resources on the LMS?",
        description="Count of accesses to course material in the learning platform. Read it "
                    "together with grades: a low count with good grades means the student works "
                    "offline; a low count with weak grades usually means little studying at all.",
        unit="views this term", actionable=True, prefer=1,
        immediate=["Check the student can actually log in - a surprising share of zeros are access problems.",
                   "Set one resource-based task a week so opening the LMS is required."],
        longterm=["Move core material into the LMS so it becomes the natural first stop.",
                  "Teach the student to use resources for retrieval practice, not just re-reading."],
    ),
    "announcements_view": dict(
        label="Announcements viewed",
        question="How many times has the student checked class announcements?",
        description="Count of announcement views. Low numbers usually mean missed deadlines "
                    "rather than laziness - the student never saw the message.",
        unit="views this term", actionable=True, prefer=1,
        immediate=["Read the week's deadlines aloud at the start of Monday's lesson.",
                   "Push announcements to a channel the student already reads."],
        longterm=["Teach the student to keep a single deadline calendar."],
    ),
    "discussion": dict(
        label="Discussion group participation",
        question="How many times has the student posted in class discussion groups?",
        description="Count of contributions to discussion boards. Explaining an idea to a peer "
                    "is one of the strongest retention activities available.",
        unit="posts this term", actionable=True, prefer=1,
        immediate=["Assign a rotating 'first responder' role so every student must post."],
        longterm=["Set open-ended prompts - factual questions kill discussion.",
                  "Build a regular peer-explanation routine into lessons."],
    ),
    "absence_days": dict(
        label="Absence days band",
        question="Has the student missed more than 7 days?",
        description="Banded absence count recorded in the xAPI study: either under 7 days or "
                    "above 7 days. Crossing 7 days is where attendance stops being incidental.",
        unit="band", actionable=True, prefer=None,
        order=["under-7", "above-7"],
        immediate=["Contact home the same week the 7-day line is crossed.",
                   "Provide a catch-up pack for every missed lesson."],
        longterm=["Find and remove the underlying barrier - transport, illness, caring duties, bullying."],
    ),
    "stage": dict(
        label="Education stage",
        question="Which education stage is the student in?",
        description="School stage (lower level, middle school, high school). Structural context "
                    "rather than something to act on.",
        unit="stage", actionable=False, prefer=None,
        order=["lowerlevel", "middleschool", "highschool"],
    ),
    "topic": dict(
        label="Course topic",
        question="Which subject is this prediction for?",
        description="Subject the student is enrolled in. Useful for spotting subjects with "
                    "systematically higher failure rates, not for judging the student.",
        unit="subject", actionable=False, prefer=None,
    ),

    # ===================================================== StudentPerf (Kaggle)
    "absences": dict(
        label="Number of absences",
        question="How many classes has the student missed this term?",
        description="Total classes missed. Across almost every education dataset this is the "
                    "single strongest behavioural predictor of failure - each missed class is "
                    "content the student must recover alone.",
        unit="classes", actionable=True, prefer=-1,
        immediate=["Call home after the 3rd absence, not the 10th.",
                   "Same-day catch-up materials so returning does not feel hopeless."],
        longterm=["Attendance contract reviewed weekly with the student and family.",
                  "Fix the practical barrier first; motivation conversations come second."],
    ),
    "study_time": dict(
        label="Weekly study time",
        question="How many hours per week does the student study outside class?",
        description="Self-reported weekly study hours outside lessons. Beyond the cohort's "
                    "upper quartile the returns flatten - how the time is used matters more "
                    "than how much of it there is.",
        unit="hours / week", actionable=True, prefer=1,
        immediate=["Timetable a fixed slot: same time, same place, four days a week.",
                   "Start with two 45-minute focused blocks rather than one long session."],
        longterm=["Teach one technique properly - retrieval practice beats re-reading.",
                  "Offer supervised study at school for students with no quiet space at home."],
    ),
    "parental_education": dict(
        label="Parental education level",
        question="What is the highest education level reached by a parent?",
        description="Highest parental education, coded 0-4. Background context. It tells you "
                    "how much explanation a family may need from the school - it says nothing "
                    "about what the student is capable of.",
        unit="level", actionable=False, prefer=None,
        levels={0: "None", 1: "High School", 2: "Some College",
                3: "Bachelor's degree", 4: "Postgraduate"},
    ),
    "parental_support": dict(
        label="Parental support",
        question="How much support does the student receive at home?",
        description="Rated support at home, coded 0-4 from none to very high. Covers practical "
                    "help (a quiet space, checking homework) as well as encouragement.",
        unit="level", actionable=True, prefer=1,
        levels={0: "None", 1: "Low", 2: "Moderate", 3: "High", 4: "Very high"},
        immediate=["Give parents one concrete nightly task - check the planner - not 'be supportive'.",
                   "Make the first call home a positive one."],
        longterm=["Weekly one-page progress summary parents can act on.",
                  "Where home support genuinely is not available, substitute a school mentor."],
    ),
    "tutoring": dict(
        label="Receiving tutoring",
        question="Is the student receiving tutoring?",
        description="Whether the student currently receives tutoring support. Important caveat: "
                    "tutoring is usually GIVEN TO students who are already struggling, so in raw "
                    "data it can look associated with failure. Compared fairly against similar "
                    "students, tutoring is associated with a higher pass rate.",
        unit="yes / no", actionable=True, prefer=1,
        levels={0: "No", 1: "Yes"},
        immediate=["Arrange tutoring now - small groups of 3-5 are close to 1:1 in effect and far cheaper.",
                   "Brief the tutor with this report's factor list so sessions target the real gap."],
        longterm=["Keep tutoring running once grades recover; withdrawing it early undoes the gain.",
                  "Tutor works on this week's class content, not a parallel syllabus."],
    ),

    # ===================================================== StudentGrades (UCI)
    "G1": dict(
        label="First-period grade (G1)",
        question="What was the student's first-period grade?",
        description="Grade from the first assessment period, on the 0-20 scale. The earliest "
                    "hard academic signal and the cheapest moment to intervene.",
        unit="0-20", actionable=True, prefer=1,
        immediate=["Book tutoring within two weeks of a weak G1 - do not wait for G2.",
                   "Have the student redo the failed paper with feedback."],
        longterm=["Weekly low-stakes quizzing so the exam is not the first real test."],
    ),
    "G2": dict(
        label="Second-period grade (G2)",
        question="What was the student's second-period grade?",
        description="Grade from the second assessment period, on the 0-20 scale. Read with G1 "
                    "it shows the trajectory: improving, flat or sliding.",
        unit="0-20", actionable=True, prefer=1,
        immediate=["Targeted revision on exactly the topics lost between G1 and G2.",
                   "Arrange academic support before the final exam, not after."],
        longterm=["Track topic-level performance so gaps are visible before they compound."],
    ),
    "failures": dict(
        label="Past class failures",
        question="How many classes has the student failed before?",
        description="Count of previously failed classes. A prior failure usually signals an "
                    "unclosed knowledge gap that keeps breaking later topics.",
        unit="classes", actionable=True, prefer=-1,
        immediate=["Diagnose the specific gap from the failed subject rather than re-teaching everything.",
                   "Plan any re-sit early in the term."],
        longterm=["Small-group remediation on prerequisite skills, twice weekly."],
    ),
    "Medu": dict(
        label="Mother's education",
        question="What is the mother's education level?",
        description="Mother's education, coded 0-4 on the UCI scale. Background context only.",
        unit="level", actionable=False, prefer=None,
        levels={0: "None", 1: "Primary (4th grade)", 2: "5th to 9th grade",
                3: "Secondary education", 4: "Higher education"},
    ),
    "Fedu": dict(
        label="Father's education",
        question="What is the father's education level?",
        description="Father's education, coded 0-4 on the UCI scale. Background context only.",
        unit="level", actionable=False, prefer=None,
        levels={0: "None", 1: "Primary (4th grade)", 2: "5th to 9th grade",
                3: "Secondary education", 4: "Higher education"},
    ),

    # ===================================================== PerfScores
    "math_score": dict(
        label="Math score",
        question="What is the student's math score?",
        description="Mathematics score out of 100. Weak numeracy tends to pull science and "
                    "technical subjects down with it.",
        unit="/100", actionable=True, prefer=1,
        immediate=["Diagnostic test to locate the exact prerequisite gap (fractions, ratio, algebra)."],
        longterm=["Fifteen minutes of daily practice beats one weekend session."],
    ),
    "science_score": dict(
        label="Science score",
        question="What is the student's science score?",
        description="Science score out of 100.",
        unit="/100", actionable=True, prefer=1,
        immediate=["Re-teach the weakest concept, then practise questions on it."],
        longterm=["Pair practical work with written explanation to lock in understanding."],
    ),
    "english_score": dict(
        label="English score",
        question="What is the student's English score?",
        description="English / language score out of 100. Weak reading fluency quietly lowers "
                    "marks in every other written subject.",
        unit="/100", actionable=True, prefer=1,
        immediate=["Explicitly teach exam command words and essay structure."],
        longterm=["Twenty minutes of daily reading at the right difficulty level."],
    ),
    "lunch": dict(
        label="Lunch programme",
        question="Is the student on standard or free/reduced lunch?",
        description="Lunch programme status, widely used as a proxy for household income. "
                    "Background context - it identifies who needs resourcing.",
        unit="category", actionable=False, prefer=None,
        order=["free/reduced", "standard"],
    ),
    "test_prep": dict(
        label="Test preparation course",
        question="Has the student completed a test-preparation course?",
        description="Whether the student completed a structured exam-preparation programme.",
        unit="category", actionable=True, prefer=None,
        order=["none", "completed"],
        immediate=["Enrol in the next prep cycle and protect the timetable slot.",
                   "If no course exists, run a six-week past-paper clinic."],
        longterm=["Build exam technique into normal teaching rather than bolting it on at the end."],
    ),

    # ===================================================== StudentFactors
    "attendance": dict(
        label="Attendance rate",
        question="What percentage of classes has the student attended?",
        description="Share of scheduled classes attended. Below roughly 90% students start "
                    "missing whole topics rather than single lessons.",
        unit="%", actionable=True, prefer=1,
        immediate=["Daily first-period check-in with one named adult.",
                   "Show the student their own attendance weekly - visibility alone moves it."],
        longterm=["Timetable the hardest subject when the student reliably attends.",
                  "Address the root cause rather than the symptom."],
    ),
    "hours_studied": dict(
        label="Hours studied per week",
        question="How many hours per week does the student study?",
        description="Weekly self-study hours outside lessons.",
        unit="hours / week", actionable=True, prefer=1,
        immediate=["Agree a realistic weekly target and track it for two weeks to get a true baseline."],
        longterm=["Two focused 45-minute blocks beat one four-hour cram.",
                  "Use timed past papers from the start of the term."],
    ),
    "motivation_level": dict(
        label="Motivation level",
        question="How motivated is the student?",
        description="Rated engagement with schooling. Low motivation is usually a symptom - of "
                    "repeated failure or of having no clear goal - rather than a root cause.",
        unit="level", actionable=True, prefer=1,
        order=["low", "medium", "high"],
        immediate=["Set one achievable short-term goal so the student experiences a win quickly.",
                   "Fortnightly check-in with a mentor the student chose."],
        longterm=["Connect the subject to a concrete post-school destination.",
                  "Give the student some genuine choice over how they are assessed."],
    ),
    "internet_access": dict(
        label="Internet access at home",
        question="Does the student have internet access at home?",
        description="Whether the student can get online at home. Without it, every online task "
                    "becomes a school-hours-only task.",
        unit="yes / no", actionable=True, prefer=1,
        order=["no", "yes"],
        immediate=["Provide offline copies of all assessed material.",
                   "Open a supervised computer room before or after school."],
        longterm=["Apply for a connectivity grant or device loan where one exists."],
    ),
    "tutoring_sessions": dict(
        label="Tutoring sessions attended",
        question="How many tutoring sessions has the student attended?",
        description="Number of tutoring sessions attended. Frequency matters more than session "
                    "length - weekly contact stops gaps compounding. As with the tutoring flag, "
                    "sessions are usually allocated to students already behind, so compare "
                    "against similar students rather than the whole cohort.",
        unit="sessions", actionable=True, prefer=1,
        immediate=["Move to weekly sessions even if each one is shorter."],
        longterm=["Keep sessions running after grades recover.",
                  "Give the tutor this report so sessions target the measured weakness."],
    ),
    "parental_involvement": dict(
        label="Parental involvement",
        question="How involved are the parents in the student's schooling?",
        description="Rated parent engagement with the student's education.",
        unit="level", actionable=True, prefer=1,
        order=["low", "medium", "high"],
        immediate=["Make the first contact positive, before any problem is raised.",
                   "Offer meeting times that work for shift workers."],
        longterm=["Regular two-way communication rather than crisis-only contact."],
    ),
    "family_income": dict(
        label="Family income band",
        question="Which income band is the family in?",
        description="Household income band. Background context - it flags which students need "
                    "resources provided, never which students to expect less from.",
        unit="band", actionable=False, prefer=None,
        order=["low", "medium", "high"],
    ),
    "previous_scores": dict(
        label="Previous exam scores",
        question="What did the student score in previous exams?",
        description="Average score in earlier exams, out of 100. Prior attainment is the "
                    "starting point, not the ceiling.",
        unit="/100", actionable=True, prefer=1,
        immediate=["Use the previous paper to identify which topics to re-teach first."],
        longterm=["Track topic-level mastery over time rather than only overall scores."],
    ),
    "sleep_hours": dict(
        label="Sleep hours per night",
        question="How many hours does the student sleep per night?",
        description="Average nightly sleep. Under-sleeping damages memory consolidation, so "
                    "study hours convert poorly into recall.",
        unit="hours / night", actionable=True, prefer=1,
        immediate=["Set a fixed wake time first - bedtime follows it."],
        longterm=["No screens for the last 45 minutes; charge the phone outside the bedroom.",
                  "Move heavy study earlier rather than extending the night."],
    ),

    # ===================================================== shared / background
    "gender": dict(
        label="Gender", question="What is the student's gender?",
        description="Recorded gender. Background context only, never a target for intervention.",
        unit="category", actionable=False, prefer=None),
    "age": dict(
        label="Age", question="How old is the student?",
        description="Age in years. Background context; older than the cohort often indicates a "
                    "previously repeated year.",
        unit="years", actionable=False, prefer=None),
}


def meta(feat: str) -> dict:
    """Metadata for a feature, with a safe fallback for unknown columns."""
    m = FEATURE_META.get(feat)
    if m:
        return m
    pretty = str(feat).replace("_", " ").strip().capitalize()
    return dict(label=pretty, question=f"What is the student's {pretty.lower()}?",
                description=f"'{feat}' as recorded in the dataset. No curated description "
                            f"available, so its interpretation is inferred from the data alone.",
                unit="", actionable=True, prefer=None)
