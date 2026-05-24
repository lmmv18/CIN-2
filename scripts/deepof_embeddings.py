import deepof.data as dof


project_path = r"C:\Users\Usuario\Desktop\deepof_project"

video_path = r"C:\Users\Usuario\Desktop\CIN-2\new_videos"
table_path = r"C:\Users\Usuario\Desktop\CIN-2\new_csv"

video_format = ".mp4"

project = dof.Project(
    project_path=project_path,
    project_name="behavior_project",

    video_path=video_path,
    table_path=table_path,

    video_format=video_format,

    animal_ids=["animal_1"],   
)

print(" Proyecto DeepOF creado")


project.create()

print(" Datos cargados y procesados")


project.learn_embeddings(
    embedding_model="contrastive",   
    latent_dim=128,                  
    epochs=100,                       
    batch_size=256,
)

print(" Embeddings entrenados")


project.export_embeddings()

print(" Embeddings exportados")