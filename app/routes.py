from flask import Blueprint,render_template,request
from app.utils.image_handler import process_image
from app.utils.celebrity_detector import CelebrityDetector
from app.utils.qa_engine import QAEngine
import base64

main=Blueprint("main",__name__)

celebrity_detector=CelebrityDetector()
qa_engine=QAEngine()

@main.route("/",methods=["GET","POST"])
def index():
    player_info=""
    result_img_data=""
    clean_img_data=""
    user_question=""
    answer=""
    player_name=""
    ai_engine="gemini"

    if request.method=="POST":
        ai_engine = request.form.get("ai_engine", "gemini")

        # 1. User asks a question
        if "question" in request.form:
            user_question=request.form["question"]
            player_name=request.form["player_name"]
            player_info=request.form["player_info"]
            result_img_data=request.form["result_img_data"]
            clean_img_data=request.form.get("clean_img_data", "")
            
            answer=qa_engine.ask_about_celebrity(player_name,user_question)

        # 2. User clicks the main "Detect" button from the top
        else:
            image_file = request.files.get("image")
            prior_clean_data = request.form.get("clean_img_data", "")

            # If there's a fresh newly uploaded picture
            if image_file and image_file.filename != "":
                try:
                    img_bytes, clear_img_bytes, face_box = process_image(image_file)
                    if face_box is not None:
                        result_img_data = base64.b64encode(img_bytes).decode()
                        clean_img_data = base64.b64encode(clear_img_bytes).decode()
                        player_info, player_name = celebrity_detector.identify(clear_img_bytes, engine=ai_engine)
                        if not player_info or player_info.strip() == "":
                            player_info = "Error: Could not identify celebrity. Please try another image."
                    else:
                        player_info = "No face detected. Please try another image."
                except Exception as e:
                    player_info = f"Error processing image: {str(e)}"
                    result_img_data = ""

            # If there's no new picture BUT they clicked Detect (Re-Detecting with new model via the top button)
            elif prior_clean_data:
                clean_img_data = prior_clean_data
                result_img_data = request.form.get("result_img_data", "")
                try:
                    clean_img_bytes = base64.b64decode(clean_img_data)
                    player_info, player_name = celebrity_detector.identify(clean_img_bytes, engine=ai_engine)
                except Exception as e:
                    player_info = f"Error changing model: {str(e)}"

    return render_template(
        "index.html",
        player_info=player_info,
        result_img_data=result_img_data,
        clean_img_data=clean_img_data,
        user_question=user_question,
        answer=answer,
        player_name=player_name,
        ai_engine=ai_engine
    )