import cv2
import smtplib
import os
import json
import time
import numpy as np
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

class SurveillanceSystem:
    """Système de surveillance complet avec email"""
    
    def __init__(self, max_images=5, save_interval=2, email_enabled=True):  # EMAIL ACTIVÉ
        print("="*60)
        print("SYSTÈME DE SURVEILLANCE AVEC EMAIL")
        print("="*60)
        
        # Paramètres
        self.MAX_IMAGES = max_images
        self.SAVE_INTERVAL = save_interval
        self.RESET_TIMEOUT = 10
        self.EMAIL_COOLDOWN = 30
        
        # Détecteur de visages
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        if self.face_cascade.empty():
            print("Erreur: Détecteur de visages non chargé")
            exit(1)
        
        # État du système
        self.images_saved = 0
        self.last_save_time = 0
        self.last_detection_time = 0
        self.in_session = False
        self.session_start = 0
        
        # Statistiques
        self.stats = {
            'total_detections': 0,
            'faces_detected': 0,
            'images_saved_total': 0,
            'sessions': 0,
            'emails_sent': 0,
            'start_time': time.time()
        }
        
        # CONFIGURATION EMAIL (À MODIFIER ICI)
        self.email_enabled = email_enabled
        if email_enabled:
            # REMPLACEZ CES VALEURS PAR LES VÔTRES 
            self.email_sender = "email@gmail.com"  # VOTRE EMAIL GMAIL
            self.email_password = "nawq uogd nktt ygva"  # MOT DE PASSE D'APPLICATION
            self.email_receiver = "destinataire@gmail.com"  # DESTINATAIRE
            
            # Vérification configuration
            if self.email_sender == "votreemail@gmail.com":
                print("ERREUR: Vous devez configurer votre email!")
                print("   Modifiez les lignes 45-47 avec vos informations")
                print("   L'email sera désactivé pour éviter les erreurs")
                self.email_enabled = False
            else:
                print("Email: ACTIVÉ et configuré")
                print(f"   Expéditeur: {self.email_sender}")
                print(f"   Destinataire: {self.email_receiver}")
        else:
            print("Email: DÉSACTIVÉ")
        
        # Création dossiers
        self.setup_storage()
        
        print(f"Images max/session: {self.MAX_IMAGES}")
        print(f"Intervalle: {self.SAVE_INTERVAL}s")
        print("="*60)
        print("Système initialisé!\n")
    
    def setup_storage(self):
        """Crée la structure de dossiers"""
        self.base_dir = "surveillance_data"
        self.images_dir = os.path.join(self.base_dir, "images")
        self.metadata_dir = os.path.join(self.base_dir, "metadata")
        
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        print(f"Données: {os.path.abspath(self.base_dir)}")
    
    def detect_faces(self, frame):
        """Détecte les visages"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            return faces
            
        except Exception as e:
            print(f"Erreur détection: {e}")
            return np.array([])
    
    def save_image(self, frame, faces):
        """Sauvegarde une image et envoie un email"""
        try:
            current_time = time.time()
            
            # Vérifications
            if len(faces) == 0:
                return False
            
            if self.images_saved >= self.MAX_IMAGES:
                return False
            
            if current_time - self.last_save_time < self.SAVE_INTERVAL:
                return False
            
            # Créer nom de fichier
            date_str = datetime.now().strftime("%Y-%m-%d")
            daily_dir = os.path.join(self.images_dir, date_str)
            os.makedirs(daily_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}_{self.images_saved+1}of{self.MAX_IMAGES}.jpg"
            image_path = os.path.join(daily_dir, filename)
            
            # Sauvegarder image
            cv2.imwrite(image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            # Sauvegarder métadonnées
            self.save_metadata(image_path, faces)
            
            # Mettre à jour compteurs
            self.images_saved += 1
            self.last_save_time = current_time
            self.stats['images_saved_total'] += 1
            
            print(f"{self.images_saved}/{self.MAX_IMAGES}] {filename}")
            
            # ENVOYER EMAIL (première image seulement)
            if self.email_enabled and self.images_saved == 1:
                email_success = self.send_email(image_path, len(faces))
                if email_success:
                    print(f"   Email envoyé à {self.email_receiver}")
                else:
                    print(f"   Échec envoi email")
            
            return True
            
        except Exception as e:
            print(f" Erreur sauvegarde: {e}")
            return False
    
    def send_email(self, image_path, face_count):
        """Envoie un email avec l'image capturée"""
        if not self.email_enabled:
            return False
        
        try:
            # Vérifier cooldown
            current_time = time.time()
            if hasattr(self, 'last_email_time'):
                if current_time - self.last_email_time < self.EMAIL_COOLDOWN:
                    return False
            
            # Préparer message
            msg = MIMEMultipart()
            msg['From'] = self.email_sender
            msg['To'] = self.email_receiver
            msg['Subject'] = f"Surveillance: {face_count} personne(s) détectée(s)"
            
            # Corps du message
            body = f"""
             SYSTÈME DE SURVEILLANCE AUTOMATIQUE
            
            Date: {datetime.now().strftime('%d/%m/%Y')}
            Heure: {datetime.now().strftime('%H:%M:%S')}
            
            DÉTECTIONS:
            • Personnes détectées: {face_count}
            • Session: #{self.stats['sessions']}
            • Image: {self.images_saved}/{self.MAX_IMAGES} de la session
            
            L'image capturée est jointe à cet email.
            
            Le système continuera à surveiller et capturer
               jusqu'à {self.MAX_IMAGES} images maximum.
            
            --------------------------------
            Notification automatique
            Système de surveillance intelligent
            """
            
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            # Joindre l'image
            with open(image_path, 'rb') as f:
                img_data = f.read()
            
            image = MIMEImage(img_data, name=os.path.basename(image_path))
            msg.attach(image)
            
            # Envoyer via Gmail
            server = smtplib.SMTP('smtp.gmail.com', 587, timeout=10)
            server.starttls()
            server.login(self.email_sender, self.email_password)
            server.send_message(msg)
            server.quit()
            
            # Mettre à jour statistiques
            self.stats['emails_sent'] += 1
            self.last_email_time = current_time
            
            return True
            
        except smtplib.SMTPAuthenticationError:
            print(" ERREUR AUTHENTIFICATION GMAIL")
            print("   Vérifiez:")
            print("   1. Email et mot de passe sont corrects")
            print("   2. Vous avez activé l'authentification à 2 facteurs")
            print("   3. Vous utilisez un mot de passe d'application")
            return False
            
        except Exception as e:
            print(f"Erreur envoi email: {type(e).__name__}: {e}")
            return False
    
    def save_metadata(self, image_path, faces):
        """Sauvegarde les métadonnées"""
        try:
            faces_data = []
            for i, (x, y, w, h) in enumerate(faces):
                faces_data.append({
                    'id': i,
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h)
                })
            
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'image_path': image_path,
                'faces_detected': len(faces),
                'faces': faces_data,
                'session': self.stats['sessions'],
                'image_number': f"{self.images_saved}/{self.MAX_IMAGES}",
                'email_sent': self.email_enabled and self.images_saved == 1
            }
            
            metadata_filename = os.path.basename(image_path).replace('.jpg', '.json')
            date_str = datetime.now().strftime("%Y-%m-%d")
            metadata_dir = os.path.join(self.metadata_dir, date_str)
            os.makedirs(metadata_dir, exist_ok=True)
            
            metadata_path = os.path.join(metadata_dir, metadata_filename)
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f" Erreur métadonnées: {e}")
    
    def update_session_state(self, faces_detected):
        """Gère l'état de la session"""
        current_time = time.time()
        
        if faces_detected > 0:
            self.last_detection_time = current_time
            
            if not self.in_session:
                self.in_session = True
                self.images_saved = 0
                self.session_start = current_time
                self.stats['sessions'] += 1
                print(f"\n SESSION #{self.stats['sessions']} DÉMARRÉE")
                print(f"   - Max {self.MAX_IMAGES} images")
                print(f"   - Email: {'OUI' if self.email_enabled else 'NON'}")
        
        if self.in_session:
            if faces_detected == 0:
                if current_time - self.last_detection_time > self.RESET_TIMEOUT:
                    self.end_session("timeout")
            
            if self.images_saved >= self.MAX_IMAGES:
                self.end_session("limit_reached")
    
    def end_session(self, reason=""):
        """Termine la session"""
        if self.in_session:
            duration = time.time() - self.session_start
            print(f"\n Session #{self.stats['sessions']} terminée")
            print(f"   • Durée: {duration:.1f}s")
            print(f"   • Images: {self.images_saved}/{self.MAX_IMAGES}")
            print(f"   • Emails envoyés: {1 if self.email_enabled and self.images_saved > 0 else 0}")
            self.in_session = False
            self.images_saved = 0
    
    def process_frame(self, frame):
        """Traite une frame"""
        faces = self.detect_faces(frame)
        faces_detected = len(faces)
        
        if faces_detected > 0:
            self.stats['total_detections'] += 1
            self.stats['faces_detected'] += faces_detected
        
        self.update_session_state(faces_detected)
        
        if self.in_session and faces_detected > 0:
            self.save_image(frame, faces)
        
        return faces
    
    def draw_interface(self, frame, faces):
        """Dessine l'interface"""
        display = frame.copy()
        
        for (x, y, w, h) in faces:
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Informations
        info = [
            f"Visages: {len(faces)}",
            f"Session: #{self.stats['sessions']}",
            f"Images: {self.images_saved}/{self.MAX_IMAGES}",
            f"Email: {'ON' if self.email_enabled else 'OFF'}"
        ]
        
        y_pos = 30
        for line in info:
            cv2.putText(display, line, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_pos += 30
        
        return display
    
    def show_statistics(self):
        """Affiche les stats"""
        elapsed = time.time() - self.stats['start_time']
        
        print("\n" + "="*60)
        print(" STATISTIQUES")
        print("="*60)
        print(f"Temps: {elapsed:.1f}s")
        print(f"Sessions: {self.stats['sessions']}")
        print(f"Détections: {self.stats['total_detections']}")
        print(f"Images sauvegardées: {self.stats['images_saved_total']}")
        print(f"Emails envoyés: {self.stats['emails_sent']}")
        
        if self.email_enabled:
            print(f"\n CONFIGURATION EMAIL:")
            print(f"   Expéditeur: {self.email_sender}")
            print(f"   Destinataire: {self.email_receiver}")
        
        print("="*60)

def main():
    """Programme principal avec email"""
    
    # CONFIGURATION
    MAX_IMAGES = 5
    SAVE_INTERVAL = 2
    EMAIL_ENABLED = True  # EMAIL ACTIVÉ
    
    print("\n" + "="*70)
    print(" SYSTÈME DE SURVEILLANCE AVEC NOTIFICATION EMAIL")
    print("="*70)
    print(f"• Maximum {MAX_IMAGES} images par session")
    print(f"• Intervalle: {SAVE_INTERVAL} secondes")
    print(f"• Email: {'ACTIVÉ' if EMAIL_ENABLED else 'DÉSACTIVÉ'}")
    
    if EMAIL_ENABLED:
        print("\n INSTRUCTIONS POUR GMAIL:")
        print("1. Allez sur: https://myaccount.google.com/security")
        print("2. Activez 'Validation en deux étapes'")
        print("3. Cliquez sur 'Mots de passe d'application'")
        print("4. Créez un nouveau mot de passe pour 'Autre'")
        print("5. Utilisez ce mot de passe dans le code")
        print("="*70)
    
    # Initialisation
    system = SurveillanceSystem(
        max_images=MAX_IMAGES,
        save_interval=SAVE_INTERVAL,
        email_enabled=EMAIL_ENABLED
    )
    
    # Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(" Erreur webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n Commandes: [s]=stats, [p]=pause, [r]=reset, [q]=quitter")
    print("\n Placez-vous devant la caméra...")
    
    cv2.namedWindow('Surveillance + Email', cv2.WINDOW_NORMAL)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            faces = system.process_frame(frame)
            display = system.draw_interface(frame, faces)
            cv2.imshow('Surveillance + Email', display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n Arrêt")
                break
            elif key == ord('s'):
                system.show_statistics()
            elif key == ord('p'):
                print("\n Système en pause - Appuyez sur 'p' pour reprendre")
                while True:
                    key2 = cv2.waitKey(1) & 0xFF
                    if key2 == ord('p'):
                        print("▶️  Reprise")
                        break
                    elif key2 == ord('q'):
                        print("\n Arrêt depuis le mode pause")
                        cap.release()
                        cv2.destroyAllWindows()
                        return
    
    except KeyboardInterrupt:
        print("\n\n Interrompu")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*70)
        print(" RAPPORT FINAL")
        system.show_statistics()
        
        print(f"\n Données: {os.path.abspath(system.base_dir)}")
        print("Terminé!")

if __name__ == "__main__":
    # Avertissement configuration
    print("  N'OUBLIEZ PAS DE CONFIGURER VOS IDENTIFIANTS GMAIL!")
    print("   Modifiez les lignes 45-47 avec vos informations")
    
    input("\nAppuyez sur Entrée pour continuer (ou Ctrl+C pour annuler)...")
    main()
