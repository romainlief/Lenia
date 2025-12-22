from launcher.launcher import Launcher


if __name__ == "__main__":
    try:
        launcher = Launcher()
        launcher.launch()
    except KeyboardInterrupt:
        print("Simulation interrompue par l'utilisateur.")
    except Exception as e:
        print(f"Une erreur est survenue : {e}")
