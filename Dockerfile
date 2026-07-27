FROM nvcr.io/nvidia/isaac-lab:2.3.2

ENV ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    SKILLMIMIC_IN_CONTAINER=1 \
    PYTHONPATH=/workspace/skillmimic-lab

WORKDIR /workspace/skillmimic-lab
COPY . .

ENTRYPOINT ["bash", "scripts/run_isaaclab.sh"]
CMD ["smoke"]
